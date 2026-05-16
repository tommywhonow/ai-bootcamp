import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report 

# 1. Create data
np.random.seed(42)
n=500
age         = np.random.randint(22, 60, n)
income      = np.random.randint(20000, 100000, n)
education   = np.random.randint(8, 22, n)
experience  = np.random.randint(0, 20, n)

hired = (0.3 * education + 0.5 * experience + np.random.normal(0, 2, n) > 10).astype(int)

df = pd.DataFrame({
    'age': age, 'income': income,
    'education': education, 'experience': experience,
    'hired': hired
})

feature_names = ['age', 'income', 'education', 'experience']
X = df[feature_names].values
y = df['hired'].values

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state = 42
)

# 3. Logistic Regression - needs scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)
acc_lr = accuracy_score(y_test, y_pred_lr)

# 4. Random Forest - NO scaling needed
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

# 5. Compare accuracy
print(f"Logistic Regression accuracy: {acc_lr:.3f}")
print(f"Random Forest accuracy:       {acc_rf:.3f}")
winner = "Random Forest" if acc_rf > acc_lr else "Logistic Regression"

# 6. Feature importance
print("\nFeature importane (rndom Forest):")
importances = rf_model.feature_importances_
for i in importances.argsort()[::-1]:
    print(f"    {feature_names[i]}: {importances[i]:.3f}")

# 7. Classification reports
print("\nLogistic Regression report: ")
print(classification_report(y_test, y_pred_rf,
                            target_names=["not hired", "hired"]))

print("Random forest report:")
print(classification_report(y_test, y_pred_lr,
                            target_names=["not hired", "hired"]))

# 8. Visualise
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

models   = ["Logistic Regression", "Random Forest"]
accuracy = [acc_lr, acc_rf]
axes[0].bar(models, accuracy, color=["#4C72B0", "#DD8542"])
axes[0].set_ylim(0.7, 1.0)
axes[0].set_title("Model Accuracy Comparision")
axes[0].set_ylabel("Accuracy")
for i, v in enumerate(accuracy):
    axes[0].text(i, v +0.005, f"{v:.3f}", ha="center", fontweight="bold")

sorted_idx = importances.argsort()[::-1]
axes[1].bar(
    [feature_names[i] for i in sorted_idx],
    [importances[i]   for i in sorted_idx],
    color="#DD8452"
)

axes[1].set_title("Random Forest Feature Importance")
axes[1].set_ylabel("Importance")

plt.tight_layout()
plt.savefig("day11_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nChart saved as day11_results.png")