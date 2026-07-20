"""
============================================================
 Dataset Organizer & Verifier  (pattern-matching version)
 Project: Evolutionary GNN for Drug Sensitivity Classification
 Desc   : Verifies all downloaded datasets are in the right
          folders using flexible glob patterns — so renamed
          / dated filenames (e.g. "screened_compounds_rel_8.5 (1).csv")
          are still detected correctly.
============================================================
"""

import os
import glob

BASE_DIR = os.path.join(os.path.dirname(__file__), '..', '..')
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')


# ── Expected files defined as glob PATTERNS, not exact names ───────────
EXPECTED = {
    "RNA-Seq Expression": {
        "folder": "expression",
        "patterns": [
            ("rnaseq_merged*.zip",
             "RNA-Seq TPM expression matrix — cell lines x genes"),
        ]
    },
    "Drug Response (IC50)": {
        "folder": "drug_response",
        "patterns": [
            ("GDSC2*.*",
             "IC50, AUC, Z-score per (cell line, drug) pair"),
        ]
    },
    "Drug Annotations": {
        "folder": "drug_annotations",
        "patterns": [
            ("screened_compounds*.csv",
             "Drug names, PubChem CIDs, target annotations"),
            ("drug_smiles.csv",
             "SMILES strings -- run download_smiles.py to generate"),
            ("drug_fingerprints.csv",
             "2048-bit Morgan fingerprints -- run generate_fingerprints.py"),
        ]
    },
    "Cell Line Metadata": {
        "folder": "cell_line_metadata",
        "patterns": [
            ("model_list*.csv",
             "Cancer type, tissue, identifiers per cell line"),
        ]
    },
    "PPI Network (STRING)": {
        "folder": "ppi_network",
        "patterns": [
            ("9606.protein.links*.txt.gz",
             "Gene-gene interaction edges with confidence scores"),
            ("9606.protein.info*.txt.gz",
             "Maps STRING protein IDs to gene names (HGNC symbols)"),
        ]
    },
}


def check_file_size(filepath):
    size = os.path.getsize(filepath)
    if size > 1_000_000_000:
        return f"{size/1e9:.1f} GB"
    elif size > 1_000_000:
        return f"{size/1e6:.1f} MB"
    elif size > 1_000:
        return f"{size/1e3:.1f} KB"
    return f"{size} B"


def find_match(folder, pattern):
    """Return the first file in `folder` matching the glob `pattern`, or None."""
    matches = glob.glob(os.path.join(folder, pattern))
    return matches[0] if matches else None


def main():
    print("\n" + "="*65)
    print("  DATASET VERIFICATION REPORT")
    print("  Project: Evolutionary GNN for Drug Sensitivity")
    print("="*65)

    total_patterns = 0
    found_count    = 0
    missing        = []
    found_paths    = {}   # category -> {pattern: actual_filepath}

    for category, info in EXPECTED.items():
        folder = os.path.join(DATA_DIR, info['folder'])
        os.makedirs(folder, exist_ok=True)

        print(f"\n📁  {category}")
        print(f"    Folder: data/raw/{info['folder']}/")

        for pattern, description in info['patterns']:
            total_patterns += 1
            match = find_match(folder, pattern)

            if match:
                found_count += 1
                size = check_file_size(match)
                actual_name = os.path.basename(match)
                print(f"    [OK]  {actual_name}  ({size})")
                print(f"          matched pattern: {pattern}")
                print(f"          {description}")
                found_paths.setdefault(category, {})[pattern] = match
            else:
                print(f"    [MISSING]  no file matching: {pattern}")
                print(f"               {description}")
                missing.append((info['folder'], pattern))

    print("\n" + "="*65)
    print(f"  SUMMARY: {found_count} / {total_patterns} files found")
    print("="*65)

    if not missing:
        print("\n  All datasets present! You can proceed to preprocessing.")
    else:
        print(f"\n  {len(missing)} file(s) missing:\n")
        for folder, pattern in missing:
            print(f"     - Need a file matching '{pattern}'")
            print(f"       inside  data/raw/{folder}/\n")

        auto_gen = ['drug_smiles.csv', 'drug_fingerprints.csv']
        missing_auto = [p for _, p in missing if p in auto_gen]
        if missing_auto:
            print("  TIP: Auto-generated files -- run these scripts:")
            print("       python src/preprocessing/download_smiles.py")
            print("       python src/preprocessing/generate_fingerprints.py")

    print("="*65 + "\n")
    return found_paths


if __name__ == '__main__':
    main()