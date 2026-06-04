import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

#1. Create data
np.random.seed(42)
n = 500
age = np.random.randint(22, 60, n)
income = np.random.randint(20000, 100000, n)
education = np.random.randint(8, 22, n)
experience = np.random.randint(0, 20, n)

hired = ( 0.3 * education + 0.5 * experience 
         + np.random.normal(0, 2, n) > 10).astype(int)

df = pd.DataFrame({
    'age': age, 'income': income,
    'education': education,
    'experience': experience,
    'hired': hired
})

feature_names = ['age', 'income', 'education', 'experience']
X = df[feature_names].values
y = df['hired'].values

#2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#3. Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
lr_model = LogisticRegression()
lr_model.fit(X_train_scaled, y_train)
acc_lr = accuracy_score(y_test, lr_model.predict(X_test_scaled))

#4. Random Forest(No Scaling)
rf_model = RandomForestClassifier(n_estimators= 100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
acc_rf = accuracy_score(y_test, rf_model.predict(X_test))

#5. XGBoost default
xgb_model = XGBClassifier(
    n_estimators=100,
    random_state=42,
    eval_metric='logloss',
    verbosity=0
)
xgb_model.fit(X_train, y_train)
acc_xgb = accuracy_score(y_test, xgb_model.predict(X_test))

#6. XGBoost tuned with GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.05, 0.1],
}
grid_search = GridSearchCV(
    estimator=XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
print('Tuning XGBoost... this takes ~30 seconds')
grid_search.fit(X_train, y_train)
print(f'Bestparams: {grid_search.best_params_}')
print(f'Best CV score: {grid_search.best_score_:.3f}')
best_xgb = grid_search.best_estimator_
acc_tuned = accuracy_score(y_test, best_xgb.predict(X_test))

#7. Compare all models
print("\n--- MODEL COPARISON ---")
results = {
    'Logistic Regression': acc_lr,
    'Random Forest':       acc_rf,
    'XGBoost (default)':   acc_xgb,
    'XGBoost (tuned)':     acc_tuned,
}
for name, acc in results.items(): 
    print(f'{name:25s}: {acc:.3f}')

winner = max(results, key=lambda k: results[k])
print(f'\nWinner: {winner}')

#8. Feature importance
print('\nFeature importance(XGBoost tuned):')
importances = best_xgb.feature_importances_
for i in importances.argsort()[::-1]:
    print(f'   {feature_names[i]}: {importances[i]:.3f}')

#9. Visualize
fig, axes = plt.subplots(1, 2, figsize=(12,5))

model_names = list(results.keys())
accuracies  = list(results.values())
colors_bar = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
bars = axes[0].bar(model_names, accuracies, color=colors_bar)
axes[0].set_ylim(0.7,1.0)
axes[0].set_title('All Model Accuracy Comparison')
axes[0].set_ylabel('Accuracy')
for bar, acc in zip(bars, accuracies):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        acc + 0.005, f'{acc: .3f}',
        ha='center', fontweight='bold', fontsize=9
    )

sorted_idx = importances.argsort()[::-1]
axes[1].bar(
    [feature_names[i] for i in sorted_idx],
    [importances[i] for i in sorted_idx],
    color='#C44E52'
)
axes[1].set_title('XGBoost Feature Importance')
axes[1].set_ylabel('Importance')

plt.tight_layout()
plt.savefig('day12_results.png', dpi=150, bbox_inches='tight')
plt.show()
print('\nChart saved as day12_results.png')
