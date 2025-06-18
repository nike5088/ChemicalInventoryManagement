from datetime import datetime, timedelta

from utils import load_datasets
from preprocess import preprocess_data
from ml.train import train_model
from ml.predict import evaluate_model, predict
import pandas as pd

def debug_predict():
    # Expecting answer of 3
    input_data = {
        "Hazard Class Encoded": 8,
        "Lab Number": 1,
        "name ID": 26,
        "Group Encoded": 1
    }

    # Expecting answer of 3
    input_data2 = {
        "Hazard Class Encoded": 4,
        "Lab Number": 1,
        "name ID": 26,
        "Group Encoded": 1
    }

    # Expecting answer of 1
    input_data3 = {
        "Hazard Class Encoded": 1,
        "Lab Number": 1,
        "name ID": 26,
        "Group Encoded": 1
    }

    # Expecting answer of 20
    input_data4 = {
        "Hazard Class Encoded": 9,
        "Lab Number": 4,
        "name ID": 24,
        "Group Encoded": 4
    }

    # Encode and predict storage location
    try:
        prediction = predict(input_data)
        prediction2 = predict(input_data2)
        prediction3 = predict(input_data3)
        prediction4 = predict(input_data4)
        print(f"\nPrediction: {prediction[0]}    answer: 3")
        print(f"\nPrediction: {prediction2[0]}    answer: 3")
        print(f"\nPrediction: {prediction3[0]}    answer: 1")
        print(f"\nPrediction: {prediction4[0]}    answer: 20")

    except Exception as e:
        print(f"Error in predicting storage location: {e}")

def expired_chemicals():
    # Inventory load out
    inventory_df, storage_df, hazard_df, owner_df = load_datasets()
    # Convert 'Received Date' column to datetime
    inventory_df['Received Date'] = pd.to_datetime(inventory_df['Received Date'], errors='coerce')

    # Adding 5-year difference
    eight_year_mark = datetime.now() - timedelta(days=8 * 365)

    # Filter the DataFrame for received dates past 8 years
    expired_df = inventory_df[inventory_df['Received Date'] < eight_year_mark]
    print("\n Any chemicals that were received after 8 years ago is considered expired.")

    # Check Ownership
    while True:
        ownership = input("What is the owner's initials? ").strip().upper()
        if ownership in owner_df['Initials'].unique():
            break
        else:
            print(f"A chemist with initials, {ownership}, do not own expired chemicals.")

    # Returns a list of expired chemicals per owners.
    columns_to_print = ["Chemical Name", "Hazard Classification", "Received Date"]
    expired_df = expired_df[expired_df['Initials'] == ownership]
    expired_df = expired_df.sort_values(by=['Received Date'], ascending=True)

    print("\n--- Expired Chemicals ---")
    print(expired_df[columns_to_print])
    print(f"\n{ownership} has {expired_df.shape[0]} expired chemicals.\n\n")

def display_lab_assignments():
    """Display the list of lab assignments."""
    # Load datasets
    inventory_df, storage_df, hazard_df, owner_df = load_datasets()

    # Specify columns to print from the desired dataset
    columns_to_print = ["First Name", "Last Name", "Lab Number"]
    print(owner_df[columns_to_print])

def register_chemical():
    """Prompt the user to enter chemical details and predict its storage location."""
    print("\n--- Register a New Chemical ---")
    chemical_name = input("Enter the chemical name (ex. Acetone): ").strip()

    while True:
        print("\nHazard Classification: 1) Carcinogenic  2) Caustic  3) Corrosive  4) Flammable  "
              "5) Generally Safe  6) Harmful  7) Irritant  8) Oxidizer  9) Toxic  10) Explosive")
        hazard_class_encoded = input("Enter the hazard classification (ex.4 for Flammable): ").strip()
        if hazard_class_encoded.isdigit():
            if 1 <= int(hazard_class_encoded) <= 10:
                break
            else:
                print("  Invalid input. Please enter valid hazard classification.")
        else:
            print("  Invalid input. Please enter a number.")

    while True:
        owner_id = input("Enter the owner's ID (ex. 26): ").strip()
        if owner_id.isdigit():
            if 1 <= int(owner_id) <=26:
                break
            else:
                print("  Invalid input. Please enter valid owner ID.")
        else:
            print("  Invalid input. Please enter a number.")

    print("\nLab Assignment")
    display_lab_assignments()
    while True:
        lab_number = input("Enter the lab number: ").strip()
        if lab_number.isdigit():
            if 1 <= int(lab_number) <= 7:
                break
            else:
                print("  Invalid input. Please enter valid lab number.")
        else:
            print("  Invalid input. Please enter a number.")

    print("\nList of Groups: 1)MHM  2)ASD  3)VUE  4)THICK  5)REM")
    while True:
        group_encoded = input("Enter the group number (ex. 1 for MHM): ").strip()
        if group_encoded.isdigit():
            if 1 <= int(group_encoded) <= 5:
                break
            else:
                print("  Invalid input. Please enter valid group number.")
        else:
            print("  Invalid input. Please enter a number.")

    # Prepare input for the model
    input_data = {
        "Hazard Class Encoded": hazard_class_encoded,
        "Lab Number": lab_number,
        "name ID": owner_id,
        "Group Encoded": group_encoded
    }

    # Debugging Prediction
    print("\nInput Data for Prediction:", input_data)

    # Encode and predict storage location
    try:
        prediction = predict(input_data)
        print(prediction)
        print(f"\nChemical '{chemical_name}' should be stored in: {prediction[0]}")
    except Exception as e:
        print(f"Error in predicting storage location: {e}")


def main():
    global merged_df
    print("Welcome to the Chemical Inventory Management System!")

    # Menu screen
    while True:
        print("\n\n--- Main Menu ---")
        print("1. Train the model")
        print("2. Register a new chemical")
        print("3. Debug Prediction")
        print("4. Check Expired Chemicals")
        print("5. Exit")

        choice = input("\nEnter your choice: ").strip()

        if choice == "1":
            # Load datasets
            inventory_df, storage_df, hazard_df, owner_df = load_datasets()

            # Preprocess and merge datasets
            merged_df = preprocess_data(inventory_df, storage_df, hazard_df, owner_df)

            # Train the Random Forest model
            rf_model, X_test, y_test = train_model(merged_df, use_randomized_search=True) # Default behavior

            # Evaluate the model
            evaluate_model(rf_model, X_test, y_test)

        elif choice == "2":
            # an option to re-train the model
            retrain = input("Do you want to retrain the model before predicting? (yes/no): ").strip().lower()

            if retrain == "yes":
                # Load dataset
                inventory_df, storage_df, hazard_df, owner_df = load_datasets()
                # Merge dataset
                merged_df = preprocess_data(inventory_df, storage_df, hazard_df, owner_df)
                # Train the model
                train_model(merged_df, use_randomized_search=True) # Default behavior
            elif retrain == "no":
                pass
            else:
                print("  Invalid input. Please enter 'yes' or 'no'.")
                continue

            register_chemical()

        elif choice == "3":
            debug_predict()

        elif choice == "4":
            expired_chemicals()

        elif choice == "5":
            print("Exiting the program...")
            break

        else:
            print("  Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
