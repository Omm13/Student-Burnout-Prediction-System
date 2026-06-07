import pandas as pd
import matplotlib.pyplot as plt

# Load final dataset
df = pd.read_csv("data/processed/final_dataset.csv")

print("Dataset Shape:", df.shape)
print("\nColumns:\n", df.columns)
print("\nMissing Values:\n", df.isnull().sum())

# -----------------------------
# 1. BURNOUT DISTRIBUTION
# -----------------------------
print("\nBurnout Distribution:\n", df["burnout_risk"].value_counts())

df["burnout_risk"].value_counts().plot(kind="bar")
plt.title("Burnout Risk Distribution")
plt.xlabel("Burnout Level")
plt.ylabel("Count")
plt.show()

# -----------------------------
# 2. STUDY HOURS VS BURNOUT
# -----------------------------
df.groupby("burnout_risk")["study_hours"].mean().plot(kind="bar")
plt.title("Average Study Hours by Burnout Level")
plt.ylabel("Study Hours")
plt.show()

# -----------------------------
# 3. SLEEP HOURS VS BURNOUT
# -----------------------------
df.groupby("burnout_risk")["sleep_hours"].mean().plot(kind="bar")
plt.title("Average Sleep Hours by Burnout Level")
plt.ylabel("Sleep Hours")
plt.show()

# -----------------------------
# 4. ACADEMIC PRESSURE VS BURNOUT
# -----------------------------
df.groupby("burnout_risk")["academic_pressure"].mean().plot(kind="bar")
plt.title("Academic Pressure vs Burnout")
plt.ylabel("Pressure Level")
plt.show()

# -----------------------------
# 5. CORRELATION HEATMAP
# -----------------------------
corr = df.corr(numeric_only=True)

plt.imshow(corr, cmap="coolwarm")
plt.colorbar()
plt.title("Feature Correlation Heatmap")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.show()