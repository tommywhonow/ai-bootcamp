import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Generate dataset with real signal
np.random.seed(42)
n = 500

education = np.random.randint(8, 22, n)
experience = np.random.randint(0, 20, n)
age = np.random.randint(22, 60, n)
income = np.random.normal(50000, 15000, n).round(2)

# Hired depends on education and experince
hired = (
    0.3 * education +
    0.5 * experience +
    np.random.normal(0,2,n)>10
).astype(int)

df = pd.DataFrame({
    "age": age,
    "income": income,
    "education": education,
    "experience": experience,
    "hired": hired
})

print("=== DATASET ===")
print(df.shape)
print(df["hired"].value_counts())
print(df.describe().round(2))

# Features and target
X = df[["age", "income", "education", "experience"]].values
y = df["hired"].values

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n=== SPLIT ===")
print("train:", X_train.shape)
print("test:", X_test.shape)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("\n=== RESULTS ===")
print(f"accuracy: {accuracy:.4f}")
print("\nclassification report:")
print(classification_report(y_test, y_pred, 
                            target_names=["not hired", "hired"]))

# Confusion matrix plot
#confusion matrix shows exactlsy WHERE the model makes mistakes
# not just how many
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6,5))
im = ax.imshow(cm, cmap="Blues")
plt.colorbar(im, ax=ax)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["not hired", "hired"])
ax.set_yticklabels(["not hired", "hired"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix (accuracy={accuracy:.2f})")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]),
        ha="center", va="center",
        fontsize=14, fontweight="bold",
        color="white" if cm[i, j] > cm.max()/2 else "black")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()
print("\nsaved confusion_matrix.png")

# Feature importance
feature_names = ["age", "income", "education", "experience"]
coefficients = model.coef_[0]

fig, ax = plt.subplots(figsize=(8, 5))
colors = ["steelblue" if c > 0 else "orange" for c in coefficients]
ax.bar(feature_names, coefficients, color=colors, edgecolor="white")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Coefficient")
ax.set_title("Feature Importance (Logistic Regression Coefficients)")
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()
print("saved feature_importance.png")