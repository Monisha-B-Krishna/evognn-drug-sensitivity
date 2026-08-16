
"""
config_loader.py
Detects whether running locally or on Colab, and resolves data paths accordingly.
Local paths stay relative to the repo (as before).
Colab paths point into the mounted Google Drive.
"""
import yaml
import os

def is_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if is_colab():
        drive_root = "/content/drive/MyDrive/gnn_drug_data"
        checkpoint_root = "/content/drive/MyDrive/gnn_drug_checkpoints"
        for key, value in config["paths"].items():
            if key.startswith("raw_"):
                relative = value.replace("data/raw/", "")
                config["paths"][key] = os.path.join(drive_root, "raw", relative)
                config["paths"]["processed_root"] = os.path.join(drive_root, "processed")
        config["paths"]["checkpoint_dir"] = checkpoint_root
        config["environment"] = "colab"
    else:
        config["paths"]["checkpoint_dir"] = "results/checkpoints"
        config["environment"] = "local"
        config["paths"]["processed_root"] = "data/processed"

    return config

if __name__ == "__main__":
    cfg = load_config()
    print("Environment:", cfg["environment"])
    print("Config loaded successfully. Keys:", list(cfg.keys()))
