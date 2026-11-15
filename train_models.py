
import os
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

np.random.seed(42)

n = 1000

data = pd.DataFrame({
    "study_hours": np.random.uniform(0, 10, n),
    "attendance": np.random.uniform(40, 100, n),
    "physics": np.random.uniform(0, 50, n),
    "mathematics": np.random.uniform(0, 50, n),
    "c_programming": np.random.uniform(0, 50, n),
    "ai_marks": np.random.uniform(0, 50, n),
    "electronics": np.random.uniform(0, 50, n),
    "assignment_marks": np.random.uniform(0, 50, n)
})

noise = np.random.normal(0, 4, n)

data["final_marks"] = (
    data["study_hours"] * 0.5
    + data["attendance"] * 0.05
    + data["physics"] * 0.12
    + data["mathematics"] * 0.12
    + data["c_programming"] * 0.12
    + data["ai_marks"] * 0.12
    + data["electronics"] * 0.12
    + data["assignment_marks"] * 0.20
    + noise *0.5
).clip(0, 50)

data.to_csv("data/student_data.csv", index=False)

features = [
    "study_hours",
    "attendance",
    "physics",
    "mathematics",
    "c_programming",
    "ai_marks",
    "electronics",
    "assignment_marks"
]

X = data[features]
y = data["final_marks"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=150,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Training completed!")
print("MAE:", round(mean_absolute_error(y_test, predictions), 2))
print("R2 Score:", round(r2_score(y_test, predictions), 3))

joblib.dump(model, "models/student_model.pkl")

print("New model saved successfully!")
print("Features:", list(model.feature_names_in_))
