import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------
# Load dataset
# -------------------------
if not os.path.exists("heart-disease.csv"):
    raise FileNotFoundError("Dataset not found. Ensure heart-disease.csv is in the same folder.")

df = pd.read_csv("heart-disease.csv")

# Replace missing value placeholders and convert to numeric
df.replace("?", np.nan, inplace=True)
df = df.apply(pd.to_numeric, errors="coerce")

# Inspect dataset
print(df.head())
print(df.info())
print("Missing values per column:")
print(df.isnull().sum())

# Fill remaining missing values with column median
df.fillna(df.median(numeric_only=True), inplace=True)

# Binarise target: 0 = no disease, 1 = disease
df["target"] = (df["target"] > 0).astype(int)
# -------------------------
# Features and target
# -------------------------
X = data.drop("target", axis=1)
y = data["target"]

print("\nClass distribution:")
print(y.value_counts())

# -------------------------
# Train-test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# Scaling for logistic regression
# -------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Scale the features for logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------
# Cross-validation setup
# -------------------------
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# -------------------------
# Logistic Regression
# -------------------------
log_model = LogisticRegression(max_iter=1000, random_state=42)

log_cv_scores = cross_val_score(log_model, X_train_scaled, y_train, cv=cv_strategy, scoring="accuracy")
print("\nLogistic Regression CV Accuracy: {:.3f} +/- {:.3f}".format(log_cv_scores.mean(), log_cv_scores.std()))

log_model.fit(X_train_scaled, y_train)
y_pred_log = log_model.predict(X_test_scaled)

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
print("Confusion Matrix (Logistic Regression):")
print(confusion_matrix(y_test, y_pred_log))
print("Classification Report (Logistic Regression):")
print(classification_report(y_test, y_pred_log, target_names=["No Disease", "Disease"]))

# -------------------------
# Random Forest
# -------------------------
rf_model = RandomForestClassifier(random_state=42)

rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=cv_strategy, scoring="accuracy")
print("\nRandom Forest (default) CV Accuracy: {:.3f} +/- {:.3f}".format(rf_cv_scores.mean(), rf_cv_scores.std()))

rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("Random Forest (default) Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Confusion Matrix (Random Forest default):")
print(confusion_matrix(y_test, y_pred_rf))

# -------------------------
# Tuned Random Forest with hyperparameters
# -------------------------
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
y_pred_best = best_rf.predict(X_test)

print("\nBest Parameters:", grid_search.best_params_)
print("Tuned Random Forest CV Accuracy: {:.3f}".format(grid_search.best_score_))
print("Tuned Random Forest Test Accuracy:", accuracy_score(y_test, y_pred_best))
print("Confusion Matrix (Tuned Random Forest):")
print(confusion_matrix(y_test, y_pred_best))
print("Classification Report (Tuned Random Forest):")
print(classification_report(y_test, y_pred_best, target_names=["No Disease", "Disease"]))

# -------------------------
# Confusion matrix visualisation
# -------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

models_to_plot = {
    "Logistic Regression": y_pred_log,
    "Tuned Random Forest": y_pred_best
}

for ax, (name, y_pred) in zip(axes, models_to_plot.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"])
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.suptitle("Confusion Matrices: Logistic Regression vs Tuned Random Forest", fontsize=13)
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.show()
print("Confusion matrix saved as confusion_matrices.png")
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

plt.figure()
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")
plt.gca().invert_yaxis()
plt.show()
