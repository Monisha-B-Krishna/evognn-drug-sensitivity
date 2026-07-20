"""
============================================================
 Missing SMILES Recovery Script  (Pass 3 - Final Automated Pass)
 Project: Evolutionary GNN for Drug Sensitivity Classification
 Desc   : For the ~67 drugs that survived Pass 1 (PubChem name)
          and Pass 2 (synonyms + ChEMBL), try four more strategies
          before giving up to manual lookup:

          Strategy C: PubChem AUTOCOMPLETE API
                      -> handles typos/partial matches/punctuation
                         differences (e.g. "JW 7 52 1" vs "JW-7-52-1")
          Strategy D: PubChem XREF search by RegistryID
                      -> some codenames are indexed as registry IDs
          Strategy E: ChEMBL EXACT + STRUCTURE search variants
                      -> strip hyphens/spaces, try alternate formats
          Strategy F: PubChem fuzzy/partial name search via
                      'fastidentity' and substring variants
                      (try removing trailing numbers, suffixes,
                       parentheses, "HCl"/"hydrochloride" etc.)

          Reads : drug_smiles_still_missing.csv
          Writes: drug_smiles_pass3_recovered.csv
                  drug_smiles_truly_missing.csv (final manual list)
============================================================
"""

import os
import re
import time
import pandas as pd
import requests
from tqdm import tqdm

BASE_DIR     = os.path.join(os.path.dirname(__file__), '..', '..')
ANNOT_FOLDER = os.path.join(BASE_DIR, 'data', 'raw', 'drug_annotations')

INPUT_FILE        = os.path.join(ANNOT_FOLDER, 'drug_smiles_still_missing.csv')
RECOVERED_FILE    = os.path.join(ANNOT_FOLDER, 'drug_smiles_pass3_recovered.csv')
TRULY_MISSING_FILE = os.path.join(ANNOT_FOLDER, 'drug_smiles_truly_missing.csv')

PUBCHEM_AUTOCOMPLETE_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/autocomplete/compound/{name}/json?limit=5"
)
PUBCHEM_NAME_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
    "{name}/property/CanonicalSMILES,IsomericSMILES/JSON"
)
CHEMBL_SEARCH_URL = (
    "https://www.ebi.ac.uk/chembl/api/data/molecule/search.json?q={name}"
)


def get_smiles_pubchem_direct(name: str):
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
    try:
        url = CHEMBL_SEARCH_URL.format(name=requests.utils.quote(name))
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            data = r.json()
            for mol in data.get('molecules', []):
                structures = mol.get('molecule_structures')
                if structures and structures.get('canonical_smiles'):
                    return structures['canonical_smiles']
        return None
    except Exception:
        return None


def autocomplete_suggestions(name: str):
    """PubChem's autocomplete catches partial/fuzzy name matches."""
    try:
        url = PUBCHEM_AUTOCOMPLETE_URL.format(name=requests.utils.quote(name))
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            data = r.json()
            return data.get('dictionary_terms', {}).get('compound', [])
        return []
    except Exception:
        return []


def generate_name_variants(name: str):
    """Generate alternate spellings/formats that often resolve where the raw name fails."""
    variants = set()
    base = str(name).strip()
    variants.add(base)

    # Replace hyphens with spaces and vice versa
    variants.add(base.replace('-', ' '))
    variants.add(base.replace(' ', '-'))
    variants.add(base.replace('-', ''))
    variants.add(base.replace(' ', ''))

    # Remove trailing salt/form suffixes
    for suffix in [' HCl', ' hydrochloride', ' free base', ' (free base)',
                   ' sodium', ' citrate', ' mesylate', ' sulfate']:
        if base.lower().endswith(suffix.lower()):
            variants.add(base[:-len(suffix)].strip())

    # Remove parenthetical content e.g. "Drug (analog)" -> "Drug"
    no_paren = re.sub(r'\(.*?\)', '', base).strip()
    if no_paren and no_paren != base:
        variants.add(no_paren)

    # Remove trailing numeric codes after a dash e.g. "Compound-12" -> "Compound"
    no_trailing_num = re.sub(r'[-\s]\d+$', '', base).strip()
    if no_trailing_num and no_trailing_num != base:
        variants.add(no_trailing_num)

    return [v for v in variants if v and len(v) > 1]


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] {INPUT_FILE} not found. Run recover_missing_smiles.py first.")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"[OK] Loaded {len(df)} still-missing drugs for Pass 3\n")

    recovered = []
    truly_missing = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Pass 3 recovery"):
        drug_id   = row.get('drug_id')
        drug_name = row.get('drug_name')
        smiles    = None
        method    = None

        variants = generate_name_variants(drug_name)

        # Strategy E: try each name variant on PubChem direct + ChEMBL
        for v in variants:
            smiles = get_smiles_pubchem_direct(v)
            if smiles:
                method = f"pubchem_variant:{v}"
                break
            time.sleep(0.25)
            smiles = get_smiles_chembl(v)
            if smiles:
                method = f"chembl_variant:{v}"
                break
            time.sleep(0.25)

        # Strategy C: PubChem autocomplete -> try resolved suggestion names
        if smiles is None:
            suggestions = autocomplete_suggestions(drug_name)
            time.sleep(0.25)
            for sug in suggestions[:3]:
                smiles = get_smiles_pubchem_direct(sug)
                if smiles:
                    method = f"pubchem_autocomplete:{sug}"
                    break
                time.sleep(0.25)

        if smiles:
            recovered.append({'drug_id': drug_id, 'drug_name': drug_name,
                              'smiles': smiles, 'method': method})
        else:
            truly_missing.append({'drug_id': drug_id, 'drug_name': drug_name})

    rec_df  = pd.DataFrame(recovered)
    miss_df = pd.DataFrame(truly_missing)

    rec_df.to_csv(RECOVERED_FILE, index=False)
    miss_df.to_csv(TRULY_MISSING_FILE, index=False)

    print(f"\n{'='*55}")
    print(f"[DONE] Pass 3 complete")
    print(f"  Input (still missing before) : {len(df)}")
    print(f"  Recovered in Pass 3           : {len(rec_df)}")
    print(f"  Truly missing (manual needed) : {len(miss_df)}")
    print(f"  Recovered file -> {RECOVERED_FILE}")
    print(f"  Truly missing  -> {TRULY_MISSING_FILE}")
    print('='*55)


if __name__ == '__main__':
    main()
