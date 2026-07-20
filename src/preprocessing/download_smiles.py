"""
============================================================
 Drug SMILES Downloader  (name-based lookup version)
 Project: Evolutionary GNN for Drug Sensitivity Classification
 Author : Monisha B K & Kiruba Samuel I V
 Desc   : Your screened_compounds.csv has no PubChem CID column,
          so this version looks up each drug BY NAME directly
          against the PubChem API (CID auto-resolved from name),
          then fetches the SMILES string.
          Falls back to trying SYNONYMS if the primary name fails.
============================================================
"""

import os
import glob
import time
import re
import pandas as pd
import requests
from tqdm import tqdm

# -- Paths ----------------------------------------------------------------
BASE_DIR     = os.path.join(os.path.dirname(__file__), '..', '..')
INPUT_FOLDER = os.path.join(BASE_DIR, 'data', 'raw', 'drug_annotations')
OUTPUT_FILE  = os.path.join(INPUT_FOLDER, 'drug_smiles.csv')
FAILED_LOG   = os.path.join(INPUT_FOLDER, 'drug_smiles_failed.csv')

# -- PubChem API: name -> CID -> SMILES in one call ------------------------
NAME_TO_PROPERTY_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
    "{name}/property/CanonicalSMILES,IsomericSMILES/JSON"
)


def find_compounds_file():
    matches = glob.glob(os.path.join(INPUT_FOLDER, 'screened_compounds*.csv'))
    if not matches:
        return None
    matches.sort(key=os.path.getmtime, reverse=True)
    return matches[0]


def clean_name(name: str) -> str:
    """Light cleanup so odd characters don't break the URL."""
    name = str(name).strip()
    name = re.sub(r'\s+', ' ', name)
    return name


_DEBUG_SHOWN = {'count': 0}  # show only the first few raw errors, not 621 of them

def get_smiles_by_name(drug_name: str):
    """Try to fetch SMILES directly from PubChem using the compound name."""
    try:
        url = NAME_TO_PROPERTY_URL.format(name=requests.utils.quote(drug_name))
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            props = response.json()['PropertyTable']['Properties'][0]
            # PubChem's current API returns the SMILES under the key "SMILES"
            # (and sometimes "ConnectivitySMILES"), not "IsomericSMILES" /
            # "CanonicalSMILES" as the older docs suggested. Check all variants.
            return (props.get('SMILES')
                    or props.get('IsomericSMILES')
                    or props.get('CanonicalSMILES')
                    or props.get('ConnectivitySMILES'))
        else:
            if _DEBUG_SHOWN['count'] < 3:
                print(f"\n  [DEBUG] '{drug_name}' -> HTTP {response.status_code}: "
                      f"{response.text[:150]}")
                _DEBUG_SHOWN['count'] += 1
            return None
    except Exception as e:
        if _DEBUG_SHOWN['count'] < 3:
            print(f"\n  [DEBUG] '{drug_name}' -> Exception: {type(e).__name__}: {e}")
            _DEBUG_SHOWN['count'] += 1
        return None


def get_first_synonym(synonyms_str):
    """SYNONYMS column is often comma/semicolon separated -- take first usable one."""
    if pd.isna(synonyms_str):
        return None
    for sep in [',', ';', '|']:
        if sep in str(synonyms_str):
            parts = [p.strip() for p in str(synonyms_str).split(sep) if p.strip()]
            return parts[0] if parts else None
    s = str(synonyms_str).strip()
    return s if s else None


def main():
    input_file = find_compounds_file()
    if input_file is None:
        print(f"[ERROR] No file matching 'screened_compounds*.csv' found in:")
        print(f"        {INPUT_FOLDER}")
        return

    print(f"[OK] Using compounds file: {os.path.basename(input_file)}")
    df = pd.read_csv(input_file)
    print(f"[OK] Loaded {len(df)} compounds")
    print(f"     Columns: {list(df.columns)}\n")

    name_col     = 'DRUG_NAME' if 'DRUG_NAME' in df.columns else df.columns[0]
    synonym_col  = 'SYNONYMS' if 'SYNONYMS' in df.columns else None
    id_col       = 'DRUG_ID' if 'DRUG_ID' in df.columns else None

    print(f"     Drug name column : '{name_col}'")
    print(f"     Synonym column   : '{synonym_col}'")
    print(f"     Drug ID column   : '{id_col}'\n")

    results = []
    failed  = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Fetching SMILES"):
        drug_id   = row[id_col] if id_col else None
        drug_name = clean_name(row[name_col])

        smiles      = None
        used_query  = None

        if drug_name and drug_name.lower() != 'nan':
            smiles = get_smiles_by_name(drug_name)
            used_query = drug_name
            time.sleep(0.3)

        # Fallback: try first synonym if primary name failed
        if smiles is None and synonym_col:
            alt_name = get_first_synonym(row[synonym_col])
            if alt_name:
                smiles = get_smiles_by_name(clean_name(alt_name))
                used_query = alt_name
                time.sleep(0.3)

        results.append({
            'drug_id': drug_id,
            'drug_name': row[name_col],
            'query_used': used_query,
            'smiles': smiles
        })

        if smiles is None:
            failed.append(row[name_col])

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_FILE, index=False)

    found   = out_df['smiles'].notna().sum()
    missing = out_df['smiles'].isna().sum()

    print(f"\n{'='*50}")
    print(f"[DONE]")
    print(f"  Total drugs    : {len(out_df)}")
    print(f"  SMILES found   : {found}")
    print(f"  SMILES missing : {missing}")
    print(f"  Saved to       : {OUTPUT_FILE}")

    if failed:
        failed_df = out_df[out_df['smiles'].isna()][['drug_id', 'drug_name']]
        failed_df.to_csv(FAILED_LOG, index=False)
        print(f"\n[WARN] {len(failed)} drugs had no SMILES found.")
        print(f"       Full list saved to: {FAILED_LOG}")
        print(f"       Preview:")
        for d in failed[:10]:
            print(f"         - {d}")
        if len(failed) > 10:
            print(f"         ... and {len(failed)-10} more")
    print('='*50)


if __name__ == '__main__':
    main()