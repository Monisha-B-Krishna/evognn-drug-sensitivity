"""
============================================================
 Missing SMILES Recovery Script  (Pass 2)
 Project: Evolutionary GNN for Drug Sensitivity Classification
 Desc   : For drugs that failed plain-name lookup on PubChem
          (mostly internal lab codenames like "JW-7-52-1",
          "CGP-082996"), this script tries additional strategies:

          Strategy A: Try each SYNONYM (not just the first one)
                      against PubChem name search.
          Strategy B: Try the ChEMBL API, which indexes many of
                      these pharma codenames that PubChem misses.

          Reads : drug_smiles_failed.csv + screened_compounds.csv
          Writes: drug_smiles_recovered.csv (newly found)
                  drug_smiles_still_missing.csv (truly not found)
============================================================
"""

import os
import glob
import time
import re
import pandas as pd
import requests
from tqdm import tqdm

BASE_DIR     = os.path.join(os.path.dirname(__file__), '..', '..')
ANNOT_FOLDER = os.path.join(BASE_DIR, 'data', 'raw', 'drug_annotations')

FAILED_FILE     = os.path.join(ANNOT_FOLDER, 'drug_smiles_failed.csv')
COMPOUNDS_FILE  = None  # resolved dynamically below

RECOVERED_FILE     = os.path.join(ANNOT_FOLDER, 'drug_smiles_recovered.csv')
STILL_MISSING_FILE = os.path.join(ANNOT_FOLDER, 'drug_smiles_still_missing.csv')

PUBCHEM_NAME_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
    "{name}/property/CanonicalSMILES,IsomericSMILES/JSON"
)
CHEMBL_SEARCH_URL = (
    "https://www.ebi.ac.uk/chembl/api/data/molecule/search.json"
    "?q={name}"
)


def find_compounds_file():
    matches = glob.glob(os.path.join(ANNOT_FOLDER, 'screened_compounds*.csv'))
    matches.sort(key=os.path.getmtime, reverse=True)
    return matches[0] if matches else None


def clean_name(name: str) -> str:
    return re.sub(r'\s+', ' ', str(name).strip())


def get_smiles_pubchem(name: str):
    try:
        url = PUBCHEM_NAME_URL.format(name=requests.utils.quote(name))
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            props = r.json()['PropertyTable']['Properties'][0]
            return (props.get('SMILES') or props.get('IsomericSMILES')
                    or props.get('CanonicalSMILES'))
        return None
    except Exception:
        return None


def get_smiles_chembl(name: str):
    """ChEMBL often indexes pharma codenames PubChem's name search misses."""
    try:
        url = CHEMBL_SEARCH_URL.format(name=requests.utils.quote(name))
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            data = r.json()
            molecules = data.get('molecules', [])
            for mol in molecules:
                structures = mol.get('molecule_structures')
                if structures and structures.get('canonical_smiles'):
                    return structures['canonical_smiles']
        return None
    except Exception:
        return None


def get_synonym_list(synonyms_str):
    if pd.isna(synonyms_str):
        return []
    s = str(synonyms_str)
    for sep in [',', ';', '|']:
        if sep in s:
            return [p.strip() for p in s.split(sep) if p.strip()]
    return [s.strip()] if s.strip() else []


def main():
    if not os.path.exists(FAILED_FILE):
        print(f"[ERROR] {FAILED_FILE} not found.")
        print("        Run download_smiles.py first.")
        return

    failed_df = pd.read_csv(FAILED_FILE)
    print(f"[OK] Loaded {len(failed_df)} previously failed drugs")

    compounds_file = find_compounds_file()
    compounds_df = pd.read_csv(compounds_file) if compounds_file else None
    if compounds_df is not None:
        compounds_df = compounds_df.set_index('DRUG_ID', drop=False)

    recovered = []
    still_missing = []

    for _, row in tqdm(failed_df.iterrows(), total=len(failed_df),
                       desc="Recovering missing SMILES"):
        drug_id   = row.get('drug_id')
        drug_name = row.get('drug_name')
        smiles    = None
        method    = None

        # Strategy A: try all synonyms via PubChem
        if compounds_df is not None and drug_id in compounds_df.index:
            synonyms_raw = compounds_df.loc[drug_id, 'SYNONYMS']
            synonym_list = get_synonym_list(synonyms_raw)
        else:
            synonym_list = []

        for syn in synonym_list:
            smiles = get_smiles_pubchem(clean_name(syn))
            if smiles:
                method = f"pubchem_synonym:{syn}"
                break
            time.sleep(0.3)

        # Strategy B: try ChEMBL with the original drug name
        if smiles is None:
            smiles = get_smiles_chembl(clean_name(drug_name))
            if smiles:
                method = "chembl_name"
            time.sleep(0.3)

        # Strategy B2: try ChEMBL with each synonym too
        if smiles is None:
            for syn in synonym_list:
                smiles = get_smiles_chembl(clean_name(syn))
                if smiles:
                    method = f"chembl_synonym:{syn}"
                    break
                time.sleep(0.3)

        if smiles:
            recovered.append({'drug_id': drug_id, 'drug_name': drug_name,
                              'smiles': smiles, 'method': method})
        else:
            still_missing.append({'drug_id': drug_id, 'drug_name': drug_name})

    rec_df = pd.DataFrame(recovered)
    miss_df = pd.DataFrame(still_missing)

    rec_df.to_csv(RECOVERED_FILE, index=False)
    miss_df.to_csv(STILL_MISSING_FILE, index=False)

    print(f"\n{'='*55}")
    print(f"[DONE] Recovery pass complete")
    print(f"  Previously missing : {len(failed_df)}")
    print(f"  Recovered now      : {len(rec_df)}")
    print(f"  Still missing      : {len(miss_df)}")
    print(f"  Recovered saved to : {RECOVERED_FILE}")
    print(f"  Still-missing list : {STILL_MISSING_FILE}")
    if len(miss_df) > 0:
        print(f"\n  These {len(miss_df)} drugs likely need manual lookup")
        print(f"  (e.g. search the drug name on google + 'SMILES' or check")
        print(f"   the original publication if it's a novel/experimental compound).")
    print('='*55)


if __name__ == '__main__':
    main()
