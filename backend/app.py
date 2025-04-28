# Import necessary libraries
from fastapi import FastAPI 
from pydantic import BaseModel
import pandas as pd
import pickle
import uvicorn
import os
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'my_model.pkl')
model_path = os.path.abspath(model_path)

model = pickle.load(open(model_path, 'rb'))
print("Model loaded successfully!")

# Define the input data model
class MatchInput(BaseModel):
    batting_team: str
    bowling_team: str
    city: str
    toss_winner: str
    match_type: str
    runs_left: int
    balls_left: int
    wickets_left: int
    remaining_wickets: int
    crr: float
    rrr: float
    target: int

# Teams and Cities (optional validation support)
teams = [
    'Royal Challengers Bengaluru', 'Mumbai Indians', 'Kolkata Knight Riders',
    'Rajasthan Royals', 'Chennai Super Kings', 'Sunrisers Hyderabad',
    'Delhi Capitals', 'Punjab Kings', 'Lucknow Super Giants', 'Gujarat Titans'
]

cities = [
    'Chennai', 'Pune', 'Delhi', 'Kolkata', 'Jaipur', 'Indore',
    'Lucknow', 'Chandigarh', 'Mumbai', 'Centurion', 'Dubai',
    'Hyderabad', 'Ahmedabad', 'Abu Dhabi', 'Bengaluru',
    'Port Elizabeth', 'Kimberley', 'Ranchi', 'Sharjah', 'East London',
    'Dharamsala', 'Johannesburg', 'Durban', 'Nagpur', 'Mohali',
    'Cuttack', 'Cape Town', 'Navi Mumbai', 'Raipur', 'Guwahati',
    'Bangalore', 'Visakhapatnam', 'Bloemfontein'
]

# Prediction endpoint
@app.post("/predict")
def predict(data: MatchInput):
    try:
        # Build DataFrame matching model training format
        input_data = {
            'batting_team': [data.batting_team],
            'bowling_team': [data.bowling_team],
            'city': [data.city],
            'toss_winner': [data.toss_winner],
            'match_type': [data.match_type],
            'runs_left': [data.runs_left],
            'balls_left': [data.balls_left],
            'wickets_left': [data.wickets_left],
            'remaining_wickets': [data.remaining_wickets],
            'crr': [data.crr],
            'rrr': [data.rrr],
            'target': [data.target]
        }

        input_df = pd.DataFrame(input_data)

        # Make prediction
        prediction = model.predict_proba(input_df)[0][1]  # Probability of winning

        return {"win_probability": round(prediction * 100, 2)}

    except Exception as e:
        return {"error": str(e)}

# Optional: Run using uvicorn if directly executed
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)


from fastapi.middleware.cors import CORSMiddleware

# After app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for now)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
