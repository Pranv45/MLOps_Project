#Import necessary libraries
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
import uvicorn
import os
from fastapi.middleware.cors import CORSMiddleware
import logging
import time

# Set up logging configuration
logging.basicConfig(
    filename="app_logs.log",    # 👈 Save logs here
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode='a'  # (append mode, don't overwrite)
)

# Create a logger object
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'lr_model.pkl')
model = pickle.load(open(model_path, 'rb'))

# Define the input data model
class MatchInput(BaseModel):
    batting_team: str
    bowling_team: str
    city: str
    match_type: str
    toss_winner: str
    total_runs_x: int
    required_runs: int
    balls_left: int
    wickets_left: int

def log_and_return_error(reason: str):
    logging.error(f"Validation failed: {reason}")
    return {"error": reason}

def log_and_return_result(log_msg: str, user_msg: str):
    logging.info(log_msg)
    return {"message": user_msg}


@app.post("/predict")
def predict(data: MatchInput):
    start_time = time.time()  # ⏳ Start timing
    try:
        logging.info(f"Received prediction request: {data}")

        if data.batting_team == data.bowling_team:
            return log_and_return_error("Batting and Bowling team cannot be the same.")
        
        if data.toss_winner not in [data.batting_team, data.bowling_team]:
            return log_and_return_error("Toss winner must be either batting or bowling team.")
        
        if data.required_runs < 0:
            return log_and_return_error("Required runs cannot be negative.")
        
        if not (0 <= data.balls_left <= 120):
            return log_and_return_error("Balls left must be between 0 and 120.")
        
        if not (0 <= data.wickets_left <= 10):
            return log_and_return_error("Wickets left must be between 0 and 10.")
        
        if data.required_runs > data.total_runs_x:
            return log_and_return_error("Required runs cannot be greater than First Innings Total.")
        
        if data.required_runs == 0 and data.wickets_left == 0:
            return log_and_return_error("Invalid input: Required runs and wickets left cannot both be zero.")

        if data.required_runs == 0:
            return log_and_return_result(
                f"Match already won by {data.batting_team} (Required Runs = 0)",
                f"Congratulations! {data.batting_team} has already won the match!"
            )

        if data.wickets_left == 0:
            return log_and_return_result(
                f"Match already won by {data.bowling_team} (Batting team all out)",
                f"All wickets lost! {data.bowling_team} has won the match!"
            )

        if data.balls_left == 0:
            if data.required_runs == 1:
                return log_and_return_result(
                    "Match tied! Both teams have equal runs after 20 overs.",
                    "Match tied! Both teams have equal runs!"
                )
            elif data.required_runs > 0:
                return log_and_return_result(
                    f"Overs completed and {data.bowling_team} has won (target not chased).",
                    f"Overs Completed! {data.bowling_team} has won the match!"
                )

        #Derive necessary features from the data 
        current_score = data.total_runs_x - data.required_runs
        balls_bowled = 120 - data.balls_left
        overs_completed = balls_bowled / 6
        crr = current_score / overs_completed if overs_completed > 0 else 0
        rrr = (data.required_runs * 6) / data.balls_left if data.balls_left > 0 else 0

        logging.info(f"Calculated features: current_score={current_score}, CRR={crr:.2f}, RRR={rrr:.2f}")

        input_df = pd.DataFrame({
            'batting_team':  [data.batting_team],
            'bowling_team':  [data.bowling_team],
            'city':          [data.city],
            'match_type':    [data.match_type],
            'toss_winner':   [data.toss_winner],
            'total_runs_x':  [data.total_runs_x],
            'required_runs': [data.required_runs],
            'balls_left':    [data.balls_left],
            'wickets_left':  [data.wickets_left],
            'crr':           [crr],
            'rrr':           [rrr]
        })

        proba = model.predict_proba(input_df)[0][1]
        win_percent = round(proba * 100, 2)

        end_time = time.time()  # ⏳ End timing
        latency = round((end_time - start_time) * 1000, 2)  # ms

        logging.info(f"Predicted Win Probability: {win_percent}%")
        logging.info(f"Prediction completed in {latency} ms")

        return {
            "message": f"Win probability of {data.batting_team} is {win_percent}%"
        }

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
