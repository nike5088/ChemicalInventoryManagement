import pandas as pd
import os

def load_datasets():
    # File paths
    inventory_file = "../data/chemical_inventory.csv"
    storage_file = "../data/storage_locations.csv"
    hazard_file = "../data/hazard_classification.csv"
    owner_file = "../data/owner_names.csv"

    # Load datasets
    if all(os.path.exists(file) for file in [inventory_file, storage_file, hazard_file, owner_file]):
        inventory_df = pd.read_csv(inventory_file)
        storage_df = pd.read_csv(storage_file)
        hazard_df = pd.read_csv(hazard_file)
        owner_df = pd.read_csv(owner_file)
        print("\n\nDatasets loaded successfully!")
        return inventory_df, storage_df, hazard_df, owner_df
    else:
        raise FileNotFoundError("One or more datasets are missing. Please check the file paths.")
