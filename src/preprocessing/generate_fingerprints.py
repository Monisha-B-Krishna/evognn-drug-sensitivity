"""
============================================================
 Drug Fingerprint Generator
 Project: Evolutionary GNN for Drug Sensitivity Classification
 Desc   : Converts SMILES strings -> 2048-bit Morgan fingerprints
          Input : drug_smiles_final.csv (the fully-recovered, 614-drug file)
          Output: data/processed/drug_fingerprints.csv
============================================================
"""

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False
    print("RDKit not installed. Run: pip install rdkit")

# Make src/ importable so we can use the shared config loader
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config_loader import load_config


def smiles_to_fingerprint(generator, smiles: str):
    """Convert a SMILES string to a Morgan fingerprint bit vector using the
    current RDKit API (rdFingerprintGenerator), not the deprecated AllChem one."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = generator.GetFingerprint(mol)
        return list(fp)
    except Exception:
        return None


def main():
    if not RDKIT_OK:
        return

    cfg = load_config()
    input_file = cfg["paths"]["processed_smiles"]        # drug_smiles_final.csv
    output_file = cfg["paths"]["processed_fingerprints"]  # drug_fingerprints.csv
    radius = cfg["fingerprints"]["radius"]
    n_bits = cfg["fingerprints"]["n_bits"]

    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        print("Confirm drug_smiles_final.csv exists at that config-defined path.")
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} drugs from {os.path.basename(input_file)}")

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    fingerprints = []
    failed = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating fingerprints"):
        drug_name = row['drug_name']
        smiles = row['smiles']

        if pd.isna(smiles):
            fingerprints.append(None)
            failed.append(drug_name)
            continue

        fp = smiles_to_fingerprint(generator, smiles)
        fingerprints.append(fp)
        if fp is None:
            failed.append(drug_name)

    df['fingerprint'] = fingerprints
    df.to_csv(output_file, index=False)

    success = sum(1 for f in fingerprints if f is not None)
    print(f"\n{'='*50}")
    print("Fingerprints generated!")
    print(f"    Successful : {success} / {len(df)}")
    print(f"    Failed     : {len(failed)}")
    print(f"    Saved to   : {output_file}")

    if failed:
        print(f"\nFailed drugs: {failed[:5]}")
    print('='*50)


if __name__ == '__main__':
    main()
