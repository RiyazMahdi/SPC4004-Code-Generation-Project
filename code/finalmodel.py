import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------
# Load dataset
# -------------------------
df = pd.read_csv("heart-disease.csv")

# Inspect dataset
print(df.head())
print(df.info())
print(df.isnull().sum())

# -------------------------
# Features and target
# -------------------------
X = df.drop("target", axis=1)
y = df["target"]

# -------------------------
# Train-test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# Scaling for logistic regression
# -------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------
# Logistic Regression
# -------------------------
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)
y_pred_log = log_model.predict(X_test_scaled)

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
print("Confusion Matrix (Logistic Regression):")
print(confusion_matrix(y_test, y_pred_log))
print("Classification Report (Logistic Regression):")
print(classification_report(y_test, y_pred_log))

# -------------------------
# Random Forest
# -------------------------
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Confusion Matrix (Random Forest):")
print(confusion_matrix(y_test, y_pred_rf))

# -------------------------
# Tuned Random Forest
# -------------------------
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5
)

grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
y_pred_best = best_rf.predict(X_test)

print("Best Parameters:", grid_search.best_params_)
print("Tuned Random Forest Accuracy:", accuracy_score(y_test, y_pred_best))
print("Confusion Matrix (Tuned Random Forest):")
print(confusion_matrix(y_test, y_pred_best))
print("Classification Report (Tuned Random Forest):")
print(classification_report(y_test, y_pred_best))