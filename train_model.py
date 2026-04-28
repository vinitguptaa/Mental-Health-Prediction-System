import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Load dataset
df = pd.read_csv("dataset/mental_health.csv")

# Remove spaces from column names
df.columns = df.columns.str.strip()

# Remove unwanted columns
drop_cols = ["id", "City", "Profession", "Degree"]

for col in drop_cols:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

# Remove missing values
df.dropna(inplace=True)

# Encode ALL categorical columns properly
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == "object" or df[col].dtype == "str":
        df[col] = le.fit_transform(df[col].astype(str))

# Target column
target_column = "Depression"

X = df.drop(target_column, axis=1)
y = df[target_column]

print("\nFinal Data Types:\n")
print(X.dtypes)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Test on new custom data
print("\nEnter New Student Data for Prediction:")

gender = int(input("Gender (Male=1, Female=0): "))
age = float(input("Age: "))
academic_pressure = float(input("Academic Pressure (1-10): "))
work_pressure = float(input("Work Pressure (1-10): "))
cgpa = float(input("CGPA: "))
study_satisfaction = float(input("Study Satisfaction (1-10): "))
job_satisfaction = float(input("Job Satisfaction (1-10): "))
sleep_duration = int(input("Sleep Duration (0=Less than 5 hrs, 1=5-6 hrs, 2=7-8 hrs, 3=More than 8 hrs): "))
dietary_habits = int(input("Dietary Habits (0=Unhealthy, 1=Moderate, 2=Healthy): "))
suicidal_thoughts = int(input("Suicidal Thoughts (Yes=1, No=0): "))
work_study_hours = float(input("Work/Study Hours: "))
financial_stress = float(input("Financial Stress (1-10): "))
family_history = int(input("Family History of Mental Illness (Yes=1, No=0): "))

new_data = [[
    gender,
    age,
    academic_pressure,
    work_pressure,
    cgpa,
    study_satisfaction,
    job_satisfaction,
    sleep_duration,
    dietary_habits,
    suicidal_thoughts,
    work_study_hours,
    financial_stress,
    family_history
]]

# ML Prediction
prediction = model.predict(new_data)

# Rule-based score logic (same as app.py)
score = 0

if academic_pressure >= 8:
    score += 2

if work_pressure >= 8:
    score += 2

if financial_stress >= 8:
    score += 2

if sleep_duration == 0:   # Less than 5 hours
    score += 2

if suicidal_thoughts == 1:
    score += 5

# Final result based on score
if score <= 4:
    print("\nPrediction: Low Mental Wellness Risk")
    print("Suggestion: You are doing well. Maintain balance in your routine and stay connected with people.")

elif score <= 8:
    print("\nPrediction: Moderate Mental Wellness Risk")
    print("Suggestion: You may be experiencing some stress. Improve sleep, reduce overload, and take regular breaks.")

else:
    print("\nPrediction: High Mental Wellness Risk")
    print("Suggestion: Please focus on your mental well-being. Talk to a mentor, counselor, or someone you trust.")


# Prediction on test data
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {round(accuracy * 100, 2)}%")

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "model.pkl")

print("\nModel saved successfully as model.pkl")