import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, roc_auc_score, roc_curve
)

# Step 0: Ensure images directory exists
os.makedirs("images", exist_ok=True)

# Step 1: Load Dataset from Kaggle CSV
file_path = "data/data.csv"
df = pd.read_csv(file_path)

# Step 2: Clean Dataset
df.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# Step 4: Standardize Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train Model
model = LogisticRegression(max_iter=5000)
model.fit(X_train_scaled, y_train)

# Step 6: Predictions
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Step 7: Evaluation
print("----- Classification Report -----\n")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nPrecision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("ROC AUC  :", roc_auc_score(y_test, y_prob))

# Step 8: ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc_score(y_test, y_prob):.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("images/roc_curve.png")  # ← Save image
plt.close()

# Step 9: Custom Threshold
threshold = 0.3
y_pred_custom = (y_prob >= threshold).astype(int)
print(f"\n--- Custom Threshold: {threshold} ---")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_custom))
print("Precision:", precision_score(y_test, y_pred_custom))
print("Recall   :", recall_score(y_test, y_pred_custom))

# Step 10: Sigmoid Curve
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.linspace(-10, 10, 100)
sig = sigmoid(z)

plt.figure(figsize=(8, 4))
plt.plot(z, sig, label='Sigmoid Function')
plt.title("Sigmoid Curve")
plt.xlabel("z")
plt.ylabel("Sigmoid(z)")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("images/sigmoid_curve.png")  # ← Save image
plt.close()

# Step 11: Class Distribution Plot
plt.figure(figsize=(5, 4))
sns.countplot(x=y, palette="Set2")
plt.title("Target Class Distribution")
plt.xticks([0, 1], ['Benign', 'Malignant'])
plt.tight_layout()
plt.savefig("images/class_distribution.png")
plt.close()

# Step 12: Correlation Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), cmap='coolwarm', annot=False)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("images/correlation_heatmap.png")
plt.close()

# Step 13: Feature Importance (Logistic Coefficients)
coefficients = pd.Series(model.coef_[0], index=X.columns)
top_features = coefficients.abs().sort_values(ascending=False).head(10)

plt.figure(figsize=(8, 5))
top_features.sort_values().plot(kind='barh', color='teal')
plt.title("Top 10 Feature Importances (Logistic Coefficients)")
plt.xlabel("Absolute Coefficient Value")
plt.tight_layout()
plt.savefig("images/feature_importance.png")
plt.close()

# Step 14: Confusion Matrix Heatmap
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Benign', 'Malignant'],
            yticklabels=['Benign', 'Malignant'])
plt.title("Confusion Matrix Heatmap")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("images/confusion_matrix.png")
plt.close()
