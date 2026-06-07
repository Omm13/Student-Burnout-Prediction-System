import pandas as pd

def load_and_standardize():

    # Load datasets
    df1 = pd.read_csv("data/raw/student_lifestyle_dataset.csv")
    df2 = pd.read_csv("data/raw/student_depression_dataset.csv")
    df3 = pd.read_csv("data/raw/Stress Indicators Dataset for Mental Health Classification.csv")

    # -----------------------------
    # SELECT + RENAME (DF1)
    # -----------------------------
    df1 = df1[[
        "Study_Hours_Per_Day",
        "Sleep_Hours_Per_Day",
        "Social_Hours_Per_Day",
        "Physical_Activity_Hours_Per_Day",
        "GPA",
        "Stress_Level"
    ]].rename(columns={
        "Study_Hours_Per_Day": "study_hours",
        "Sleep_Hours_Per_Day": "sleep_hours",
        "Social_Hours_Per_Day": "social_hours",
        "Physical_Activity_Hours_Per_Day": "physical_activity",
        "GPA": "gpa",
        "Stress_Level": "academic_pressure"
    })

    # -----------------------------
    # SELECT + RENAME (DF2)
    # -----------------------------
    df2 = df2[[
        "Gender",
        "Age",
        "Financial Stress",
        "Study Satisfaction",
        "Work/Study Hours"
    ]].rename(columns={
        "Gender": "gender",
        "Age": "age",
        "Financial Stress": "financial_stress",
        "Study Satisfaction": "study_satisfaction",
        "Work/Study Hours": "study_hours_2"
    })

    # -----------------------------
    # SELECT (DF3)
    # -----------------------------
    df3 = df3[[
        "peer_competition",
        "relationship_stress",
        "sleep_problems",
        "irritability"
    ]]

    # -----------------------------
    # ALIGN LENGTH (IMPORTANT)
    # -----------------------------
    min_len = min(len(df1), len(df2), len(df3))

    df1 = df1.iloc[:min_len].reset_index(drop=True)
    df2 = df2.iloc[:min_len].reset_index(drop=True)
    df3 = df3.iloc[:min_len].reset_index(drop=True)

    # -----------------------------
    # MERGE
    # -----------------------------
    df = pd.concat([df1, df2, df3], axis=1)

    # -----------------------------
    # COMBINE STUDY HOURS
    # -----------------------------
    df["study_hours"] = (df["study_hours"] + df["study_hours_2"]) / 2
    df.drop(columns=["study_hours_2"], inplace=True)

    # -----------------------------
    # CLEAN BASIC
    # -----------------------------
    df.dropna(inplace=True)

    df.to_csv("data/processed/merged_dataset.csv", index=False)
    print("Merged dataset created successfully.")

    return df


if __name__ == "__main__":
    load_and_standardize()