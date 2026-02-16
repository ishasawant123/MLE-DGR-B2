#FILE NAME : PIPELINE_JOBLIB.PY
import pandas as pd
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

data =  pd.DataFrame({
    "study_hours": [1,2,3,4,5,6,7,8],
    "attendence": [55,60,65,70,75,80,85,90],
    "pass": [0,0,0,0,1,1,1,1]
})

X =  data[["study_hours", "attendence"]]
y = data["pass"]

X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size=0.25, random_state=42
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test,y_pred))

#dump means save
dump(pipeline, "pipeline.joblib")
dump(pipeline.named_steps["scaler"], "scaler.joblib")
dump(pipeline.named_steps["model"], "model.joblib")

new_data = pd.DataFrame({
    "study_hours":[6],
    "attendence": [78]
})

#option 1 : pipeline
pipeline_loaded = load("pipeline.joblib")
prediction = pipeline_loaded.predict(new_data)
print("Pipeline Prediction: ", "Pass" if prediction[0] == 1 else "Fail")

#option2: scaler + model (manual)
scaler_loaded = load("scaler.joblib")
model_loaded = load("model.joblib")

new_data_scaled = scaler_loaded.transform(new_data)
prediction = model_loaded.predict(new_data_scaled)
print("Model-only Prediction: ", "Pass" if prediction[0] == 1 else "Fail")
