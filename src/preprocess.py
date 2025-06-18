def preprocess_data(inventory_df, storage_df, hazard_df, owner_df):
    # Encode labels of Hazardous Classification column
    mapping_haz_class = {
        "Carcinogenic": 1,
        "Caustic": 2,
        "Corrosive": 3,
        "Flammable": 4,
        "Generally Safe": 5,
        "Harmful": 6,
        "Irritant": 7,
        "Oxidizer": 8,
        "Toxic": 9,
        "Explosive": 10
    }
    inventory_df["Hazard Class Encoded"] = inventory_df["Hazard Classification"].map(mapping_haz_class)

    # Encode labels of Group column
    mapping_group = {
        "MHM": 1,
        "ASD": 2,
        "VUE": 3,
        "THICK": 4,
        "REM": 5,
    }
    owner_df["Group Encoded"] = owner_df["Group"].map(mapping_group)

    # Merge datasets
    merged_df = inventory_df.merge(owner_df, on="Initials", how="left")  # Map owner initials to labs
    merged_df = merged_df.merge(hazard_df, on="Hazard Classification", how="left")  # Hazard compatibility
    merged_df = merged_df.merge(storage_df, on=["Lab Number", "Hazard Category"], how="left")  # Match Storage ID

    print("Datasets compiled successfully!\n\n")

    # Fill missing value
    merged_df.fillna("Unknown", inplace=True)
    return merged_df

def preprocess_expired(inventory_df, storage_df, hazard_df, owner_df, five_year_mark):
    # Encode labels of Hazardous Classification column
    mapping_haz_class = {
        "Carcinogenic": 1,
        "Caustic": 2,
        "Corrosive": 3,
        "Flammable": 4,
        "Generally Safe": 5,
        "Harmful": 6,
        "Irritant": 7,
        "Oxidizer": 8,
        "Toxic": 9,
        "Explosive": 10
    }
    inventory_df["Hazard Class Encoded"] = inventory_df["Hazard Classification"].map(mapping_haz_class)

    # Encode labels of Group column
    mapping_group = {
        "MHM": 1,
        "ASD": 2,
        "VUE": 3,
        "THICK": 4,
        "REM": 5,
    }
    owner_df["Group Encoded"] = owner_df["Group"].map(mapping_group)
    print(owner_df.head(20))

    # Merge datasets
    merged_df = inventory_df.merge(owner_df, on="Initials", how="left")  # Map owner initials to labs
    merged_df = merged_df.merge(hazard_df, on="Hazard Classification", how="left")  # Hazard compatibility
    merged_df = merged_df.merge(storage_df, on=["Lab Number", "Hazard Category"], how="left")  # Match Storage ID

    print("Datasets merged successfully!")
    print(merged_df.head(20))

    # Fill missing value
    merged_df.fillna("Unknown", inplace=True)
    return merged_df