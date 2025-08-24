import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# --------------------------
# Load dataset
# --------------------------
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Drop customerID
df = df.drop("customerID", axis=1)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

# Encode Yes/No
df = df.replace({"Yes": 1, "No": 0})

# One-hot encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# --------------------------
# Split into features & target
# --------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------
# Train Logistic Regression
# --------------------------
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# --------------------------
# Evaluate
# --------------------------
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --------------------------
# Save model and columns
# --------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/churn_model.pkl")
joblib.dump(X.columns, "models/churn_model_columns.pkl")  # IMPORTANT for Streamlit
print("✅ Model and columns saved in /models folder")