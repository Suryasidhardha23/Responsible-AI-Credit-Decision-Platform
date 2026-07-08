import os
import pandas as pd
import numpy as np

def fetch_and_clean_dataset():
    print("Fetching German Credit dataset from UCI...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    
    # Define original column names
    col_names = [
        "checking_status", "duration", "credit_history", "purpose", "credit_amount",
        "savings_status", "employment_status", "installment_rate", "personal_status_sex",
        "other_parties", "residence_since", "property_magnitude", "age", "other_payment_plans",
        "housing", "existing_credits", "job", "num_dependents", "own_telephone", "foreign_worker",
        "class"
    ]
    
    # Read the dataset
    df = pd.read_csv(url, sep=' ', header=None, names=col_names)
    
    # Define categorical mappings
    checking_map = {
        "A11": "<0 DM",
        "A12": "0 <= ... < 200 DM",
        "A13": ">= 200 DM",
        "A14": "no checking account"
    }
    
    history_map = {
        "A30": "no credits taken",
        "A31": "all credits paid back duly",
        "A32": "existing credits paid back duly",
        "A33": "delay in paying off in the past",
        "A34": "critical account"
    }
    
    purpose_map = {
        "A40": "car (new)",
        "A41": "car (used)",
        "A42": "furniture/equipment",
        "A43": "radio/television",
        "A44": "domestic appliances",
        "A45": "repairs",
        "A46": "education",
        "A47": "vacation",
        "A48": "retraining",
        "A49": "business",
        "A410": "others"
    }
    
    savings_map = {
        "A61": "< 100 DM",
        "A62": "100 <= ... < 500 DM",
        "A63": "500 <= ... < 1000 DM",
        "A64": ">= 1000 DM",
        "A65": "unknown/ no savings account"
    }
    
    employment_map = {
        "A71": "unemployed",
        "A72": "< 1 year",
        "A73": "1 <= ... < 4 years",
        "A74": "4 <= ... < 7 years",
        "A75": ">= 7 years"
    }
    
    housing_map = {
        "A151": "rent",
        "A152": "own",
        "A153": "for free"
    }
    
    job_map = {
        "A171": "unemployed/unskilled non-resident",
        "A172": "unskilled resident",
        "A173": "skilled employee/official",
        "A174": "management/highly qualified"
    }
    
    # Map personal status and sex
    # A91 : male : divorced/separated
    # A92 : female : divorced/separated/married
    # A93 : male : single
    # A94 : male : married/widowed
    # A95 : female : single
    def decode_personal_status_sex(code):
        if code == "A91":
            return "male", "divorced/separated"
        elif code == "A92":
            return "female", "divorced/separated/married"
        elif code == "A93":
            return "male", "single"
        elif code == "A94":
            return "male", "married/widowed"
        elif code == "A95":
            return "female", "single"
        return "unknown", "unknown"
    
    # Apply maps
    df["checking_status"] = df["checking_status"].map(checking_map)
    df["credit_history"] = df["credit_history"].map(history_map)
    df["purpose"] = df["purpose"].map(purpose_map)
    df["savings_status"] = df["savings_status"].map(savings_map)
    df["employment_status"] = df["employment_status"].map(employment_map)
    df["housing"] = df["housing"].map(housing_map)
    df["job"] = df["job"].map(job_map)
    
    # Extract sex and personal status
    decoded = df["personal_status_sex"].apply(decode_personal_status_sex)
    df["sex"] = [x[0] for x in decoded]
    df["personal_status"] = [x[1] for x in decoded]
    df.drop(columns=["personal_status_sex"], inplace=True)
    
    # Add age_group sensitive column (young: < 25, adult: >= 25)
    df["age_group"] = np.where(df["age"] < 25, "young", "adult")
    
    # Map target column (1 = Good/Approved, 2 = Bad/Rejected) -> (1, 0)
    df["target"] = np.where(df["class"] == 1, 1, 0)
    df.drop(columns=["class"], inplace=True)
    
    # Map binary indicators
    df["own_telephone"] = np.where(df["own_telephone"] == "A192", "yes", "no")
    df["foreign_worker"] = np.where(df["foreign_worker"] == "A201", "yes", "no")
    
    # Ensure folder exists and save
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/german_credit.csv", index=False)
    print("Successfully saved German Credit dataset to data/german_credit.csv!")
    print(f"Shape: {df.shape}")
    print(df.head(2))

if __name__ == "__main__":
    fetch_and_clean_dataset()
