from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

# Load pre-trained model and label encoder
with open("stacking_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

app = FastAPI()


class InputData(BaseModel):
    Time_spent_Alone: float
    Stage_fear: str  # e.g., 'Yes' or 'No'
    Social_event_attendance: float
    Going_outside: float
    Drained_after_socializing: str  # e.g., 'Yes' or 'No'
    Friends_circle_size: float
    Post_frequency: float


@app.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([data.dict()])

    if list(df.columns) != [
        "Time_spent_Alone",
        "Stage_fear",
        "Social_event_attendance",
        "Going_outside",
        "Drained_after_socializing",
        "Friends_circle_size",
        "Post_frequency"
    ]:
        return {"error": "Invalid input format."}

    pred = model.predict(df)
    return {"prediction": le.inverse_transform(pred)[0]}


# Optional: add a health check route
@app.get("/health")
def health():
    return {"status": "ok"}
