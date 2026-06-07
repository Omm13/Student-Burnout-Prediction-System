import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import shap

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("data/processed/final_dataset.csv")

print("Dataset Shape:", df.shape)

# -----------------------------
# FEATURES & TARGET
# -----------------------------
X = df.drop(["burnout_risk", "academic_pressure"], axis=1)
y = df["burnout_risk"]

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# MODEL (CatBoost)
# -----------------------------
model = CatBoostClassifier(
    iterations=200,
    depth=6,
    learning_rate=0.1,
    verbose=0
)

# -----------------------------
# TRAIN
# -----------------------------
model.fit(X_train, y_train)

# -----------------------------
# PREDICT
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
importances = model.get_feature_importance()
feature_names = X.columns

# Create dataframe
feat_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:\n", feat_df)

# Plot
plt.figure()
plt.barh(feat_df["Feature"], feat_df["Importance"])
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.gca().invert_yaxis()
plt.show()

# -----------------------------
# SHAP EXPLANATION
# -----------------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test)

joblib.dump(X.columns.tolist(), "feature_names.pkl")
# -----------------------------
# EVALUATION
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -----------------------------
# SAVE MODEL
# -----------------------------
joblib.dump(model, "data/processed/burnout_model.pkl")

print("\nModel saved as burnout_model.pkl")