"""
============================================================
 Drug Fingerprint Generator
 Project: Evolutionary GNN for Drug Sensitivity Classification
 Desc   : Converts SMILES strings → 2048-bit Morgan fingerprints
          Input : drug_smiles.csv
          Output: drug_fingerprints.csv
============================================================
"""

import os
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False
    print("❌  RDKit not installed. Run: pip install rdkit")

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR    = os.path.join(os.path.dirname(__file__), '..', '..')
INPUT_FILE  = os.path.join(BASE_DIR, 'data', 'raw', 'drug_annotations',
                           'drug_smiles.csv')
OUTPUT_FILE = os.path.join(BASE_DIR, 'data', 'raw', 'drug_annotations',
                           'drug_fingerprints.csv')


def smiles_to_fingerprint(smiles: str,
                           radius: int = 2,
                           n_bits: int = 2048) -> list | None:
    """Convert a SMILES string to a Morgan fingerprint bit vector."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol,
                                                   radius=radius,
                                                   nBits=n_bits)
        return list(fp)
    except Exception:
        return None


def main():
    if not RDKIT_OK:
        return

    if not os.path.exists(INPUT_FILE):
        print(f"❌  File not found: {INPUT_FILE}")
        print("    Run download_smiles.py first.")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"✅  Loaded {len(df)} drugs from drug_smiles.csv")

    fingerprints = []
    failed       = []

    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc="Generating fingerprints"):
        drug_name = row['drug_name']
        smiles    = row['smiles']

        if pd.isna(smiles):
            fingerprints.append(None)
            failed.append(drug_name)
            continue

        fp = smiles_to_fingerprint(smiles)
        fingerprints.append(fp)
        if fp is None:
            failed.append(drug_name)

    df['fingerprint'] = fingerprints

    # Save: drug_name, pubchem_cid, smiles, fingerprint (as list string)
    df.to_csv(OUTPUT_FILE, index=False)

    success = sum(1 for f in fingerprints if f is not None)
    print(f"\n{'='*50}")
    print(f"✅  Fingerprints generated!")
    print(f"    Successful : {success} / {len(df)}")
    print(f"    Failed     : {len(failed)}")
    print(f"    Saved to   : {OUTPUT_FILE}")

    if failed:
        print(f"\n⚠️   Failed drugs: {failed[:5]}")
    print('='*50)


if __name__ == '__main__':
    main()
