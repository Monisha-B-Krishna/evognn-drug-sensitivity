"""
============================================================
 Merge SMILES Results  (Pass 1 + 2 + 3 -> Final File)
 Project: Evolutionary GNN for Drug Sensitivity Classification
 Desc   : Combines all three recovery passes into one final
          clean drug_smiles_final.csv, flags low-confidence
          matches from Pass 3 variant-stripping for a quick
          sanity check, and builds a manual checklist for
          whatever's still truly missing.
============================================================
"""

import os
import glob
import pandas as pd

BASE_DIR     = os.path.join(os.path.dirname(__file__), '..', '..')
ANNOT_FOLDER = os.path.join(BASE_DIR, 'data', 'raw', 'drug_annotations')

PASS1_FILE         = os.path.join(ANNOT_FOLDER, 'drug_smiles.csv')
PASS2_FILE         = os.path.join(ANNOT_FOLDER, 'drug_smiles_recovered.csv')
PASS3_FILE         = os.path.join(ANNOT_FOLDER, 'drug_smiles_pass3_recovered.csv')
TRULY_MISSING_FILE = os.path.join(ANNOT_FOLDER, 'drug_smiles_truly_missing.csv')

FINAL_FILE          = os.path.join(ANNOT_FOLDER, 'drug_smiles_final.csv')
REVIEW_FILE          = os.path.join(ANNOT_FOLDER, 'drug_smiles_review_needed.csv')
MANUAL_CHECKLIST     = os.path.join(ANNOT_FOLDER, 'manual_lookup_checklist.csv')


def find_compounds_file():
    matches = glob.glob(os.path.join(ANNOT_FOLDER, 'screened_compounds*.csv'))
    matches.sort(key=os.path.getmtime, reverse=True)
    return matches[0] if matches else None


def main():
    pass1 = pd.read_csv(PASS1_FILE)
    pass2 = pd.read_csv(PASS2_FILE) if os.path.exists(PASS2_FILE) else pd.DataFrame()
    pass3 = pd.read_csv(PASS3_FILE) if os.path.exists(PASS3_FILE) else pd.DataFrame()

    # Pass 1 successes
    pass1_ok = pass1[pass1['smiles'].notna()].copy()
    pass1_ok['method'] = 'pubchem_direct_name'
    pass1_ok = pass1_ok[['drug_id', 'drug_name', 'smiles', 'method']]

    # Pass 2 successes
    if len(pass2) > 0:
        pass2_ok = pass2[pass2['smiles'].notna()][['drug_id', 'drug_name', 'smiles', 'method']]
    else:
        pass2_ok = pd.DataFrame(columns=['drug_id', 'drug_name', 'smiles', 'method'])

    # Pass 3 successes
    if len(pass3) > 0:
        pass3_ok = pass3[pass3['smiles'].notna()][['drug_id', 'drug_name', 'smiles', 'method']]
    else:
        pass3_ok = pd.DataFrame(columns=['drug_id', 'drug_name', 'smiles', 'method'])

    # Combine all
    final_df = pd.concat([pass1_ok, pass2_ok, pass3_ok], ignore_index=True)
    final_df = final_df.drop_duplicates(subset=['drug_id'], keep='first')
    final_df.to_csv(FINAL_FILE, index=False)

    # Flag Pass 3 "variant" matches for a quick manual sanity check --
    # these used name-stripping (removing HCl, trailing numbers, etc.)
    # so there's a small chance of matching a related-but-different compound.
    review_df = pass3_ok[pass3_ok['method'].str.contains('variant', na=False)].copy()
    if len(review_df) > 0:
        review_df['pubchem_verify_url'] = review_df['drug_name'].apply(
            lambda n: f"https://pubchem.ncbi.nlm.nih.gov/#query={str(n).replace(' ', '%20')}"
        )
        review_df.to_csv(REVIEW_FILE, index=False)

    # Manual checklist for the 7 truly missing
    truly_missing_df = pd.read_csv(TRULY_MISSING_FILE) if os.path.exists(TRULY_MISSING_FILE) else pd.DataFrame()
    compounds_file = find_compounds_file()

    if len(truly_missing_df) > 0 and compounds_file:
        compounds_df = pd.read_csv(compounds_file)
        checklist = truly_missing_df.merge(
            compounds_df[['DRUG_ID', 'SYNONYMS', 'TARGET', 'TARGET_PATHWAY']],
            left_on='drug_id', right_on='DRUG_ID', how='left'
        )
        checklist['pubchem_search_url'] = checklist['drug_name'].apply(
            lambda n: f"https://pubchem.ncbi.nlm.nih.gov/#query={str(n).replace(' ', '%20')}"
        )
        checklist['chemspider_search_url'] = checklist['drug_name'].apply(
            lambda n: f"https://www.chemspider.com/Search.aspx?q={str(n).replace(' ', '%20')}"
        )
        checklist['smiles_to_fill_in'] = ''
        checklist = checklist[['drug_id', 'drug_name', 'SYNONYMS', 'TARGET',
                               'TARGET_PATHWAY', 'pubchem_search_url',
                               'chemspider_search_url', 'smiles_to_fill_in']]
        checklist.to_csv(MANUAL_CHECKLIST, index=False)

    total_drugs = len(pd.read_csv(compounds_file)) if compounds_file else 'unknown'

    print("="*60)
    print("  FINAL MERGE REPORT")
    print("="*60)
    print(f"  Total drugs in dataset       : {total_drugs}")
    print(f"  Pass 1 (direct name)         : {len(pass1_ok)}")
    print(f"  Pass 2 (synonym + ChEMBL)    : {len(pass2_ok)}")
    print(f"  Pass 3 (variants+autocomplete): {len(pass3_ok)}")
    print(f"  TOTAL FOUND                  : {len(final_df)}")
    print(f"  Truly missing (manual needed): {len(truly_missing_df)}")
    print(f"\n  Final merged file -> {FINAL_FILE}")

    if len(review_df) > 0:
        print(f"\n  [REVIEW SUGGESTED] {len(review_df)} drugs matched via name-stripping")
        print(f"  (e.g. removed 'HCl', trailing numbers, etc.) -- small chance of")
        print(f"  matching a related but technically different salt/form.")
        print(f"  Quick sanity check file -> {REVIEW_FILE}")
        print(f"  (Open it, spot-check a few SMILES against the drug_name on PubChem.)")

    if len(truly_missing_df) > 0:
        print(f"\n  Manual checklist for final {len(truly_missing_df)} -> {MANUAL_CHECKLIST}")
    print("="*60)


if __name__ == '__main__':
    main()
