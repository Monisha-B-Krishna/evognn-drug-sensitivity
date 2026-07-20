"""
config_loader.py
Central place every script uses to read config.yaml.
Never hardcode a path or hyperparameter in another script — read it from here.
"""
import yaml

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    cfg = load_config()
    print("Config loaded successfully. Keys:", list(cfg.keys()))