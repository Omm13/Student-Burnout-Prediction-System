import pandas as pd

# -----------------------------
# BURNOUT LOGIC
# -----------------------------
def create_burnout_label(row):
    score = 0

    if row["study_hours"] > 8:
        score += 1

    if row["sleep_hours"] < 6:
        score += 1

    if row["academic_pressure"] > 3:
        score += 1

    if score == 3:
        return "High"
    elif score == 2:
        return "Medium"
    else:
        return "Low"


def clean_data():

    # -----------------------------
    # LOAD DATA
    # -----------------------------
    df = pd.read_csv("data/processed/merged_dataset.csv")

    print("Initial shape:", df.shape)

    # -----------------------------
    # CONVERT ACADEMIC PRESSURE
    # -----------------------------
    if df["academic_pressure"].dtype == "object":
        df["academic_pressure"] = df["academic_pressure"].map({
            "Low": 1,
            "Medium": 3,
            "High": 5
        })

    df["academic_pressure"] = pd.to_numeric(df["academic_pressure"], errors="coerce")

    # -----------------------------
    # CONVERT OTHER NUMERIC COLUMNS
    # -----------------------------
    cols_to_convert = [
        "study_hours", "sleep_hours", "social_hours",
        "physical_activity", "gpa", "financial_stress",
        "study_satisfaction", "peer_competition",
        "relationship_stress", "sleep_problems", "irritability",
        "age"
    ]

    for col in cols_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # -----------------------------
    # HANDLE MISSING VALUES
    # -----------------------------
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)

    # Fill categorical
    if "gender" in df.columns:
        df["gender"].fillna("Male", inplace=True)

    # -----------------------------
    # ENCODE GENDER
    # -----------------------------
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

    # -----------------------------
    # AGE FILTER
    # -----------------------------
    df = df[(df["age"] >= 16) & (df["age"] <= 35)]

    print("After cleaning shape:", df.shape)

    # -----------------------------
    # CREATE TARGET VARIABLE
    # -----------------------------
    df["burnout_risk"] = df.apply(create_burnout_label, axis=1)

    # -----------------------------
    # SAVE FINAL DATASET
    # -----------------------------
    df.to_csv("data/processed/final_dataset.csv", index=False)

    print("\nFinal dataset created successfully.")
    print("Dataset shape:", df.shape)
    print("\nBurnout Distribution:\n", df["burnout_risk"].value_counts())


if __name__ == "__main__":
    clean_data()