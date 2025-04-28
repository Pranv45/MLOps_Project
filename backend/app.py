from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

app = FastAPI()

model_path = 'model/ipl_model.pkl'
encoder_path = 'model/encoders.pkl'

if os.path.exists(model_path) and os.path.exists(encoder_path):
    model = joblib.load(model_path)
    encoders = joblib.load(encoder_path)
else:
    model = None
    encoders = None

class MatchInput(BaseModel):
    team1: str
    team2: str
    venue: str
    toss_winner: str

@app.post('/predict')
def predict_match(match: MatchInput):
    if model is None:
        return {"error": "Model not found. Train the model first!"}
    try:
        team1 = encoders['team1'].transform([match.team1])[0]
        team2 = encoders['team2'].transform([match.team2])[0]
        venue = encoders['venue'].transform([match.venue])[0]
        toss_winner = encoders['toss_winner'].transform([match.toss_winner])[0]

        X = [[team1, team2, venue, toss_winner]]
        pred = model.predict(X)[0]

        winner = encoders['winner'].inverse_transform([pred])[0]
        return {'predicted_winner': winner}
    except Exception as e:
        return {'error': str(e)}
