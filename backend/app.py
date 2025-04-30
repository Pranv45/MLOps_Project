# backend/app.py

import os
import pickle
import logging
import time

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse    # ← import JSONResponse
import uvicorn
from pydantic import BaseModel
import pandas as pd

# Prometheus client imports
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# ───── Logging setup ─────
logging.basicConfig(
    filename="app_logs.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode='a'
)

# ───── Prometheus metrics ─────
REQUEST_COUNT = Counter(
    "ipl_win_predictor_requests_total",
    "Total number of requests received",
    ["method", "endpoint", "http_status"]
)
REQUEST_LATENCY = Histogram(
    "ipl_win_predictor_request_latency_seconds",
    "Latency in seconds per request",
    ["method", "endpoint"]
)
ERROR_COUNT = Counter(
    "ipl_win_predictor_errors_total",
    "Total number of errors encountered",
    ["method", "endpoint"]
)

# ───── FastAPI app setup ─────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'lr_model.pkl')
model = pickle.load(open(model_path, 'rb'))


# ───── Request /predict payload schema ─────
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

# ───── Prometheus scrape endpoint ─────
@app.get("/metrics")
def metrics():
    """
    Expose Prometheus metrics.
    """
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

# ───── Prediction endpoint ─────
@app.post("/predict")
def predict(request: Request, data: MatchInput):
    start_time = time.time()
    endpoint = request.url.path
    method = request.method

    try:
        logging.info(f"Received prediction request: {data}")

        # ─── Validation checks ───
        if data.batting_team == data.bowling_team:
            resp = log_and_return_error("Batting and Bowling team cannot be the same.")
            status = 400

        elif data.toss_winner not in [data.batting_team, data.bowling_team]:
            resp = log_and_return_error("Toss winner must be either batting or bowling team.")
            status = 400

        elif data.required_runs < 0:
            resp = log_and_return_error("Required runs cannot be negative.")
            status = 400

        elif not (0 <= data.balls_left <= 120):
            resp = log_and_return_error("Balls left must be between 0 and 120.")
            status = 400

        elif not (0 <= data.wickets_left <= 10):
            resp = log_and_return_error("Wickets left must be between 0 and 10.")
            status = 400

        elif data.required_runs > data.total_runs_x:
            resp = log_and_return_error("Required runs cannot be greater than First Innings Total.")
            status = 400

        elif data.required_runs == 0 and data.wickets_left == 0:
            resp = log_and_return_error("Invalid input: Required runs and wickets left cannot both be zero.")
            status = 400

        # ─── Match-already-decided logic ───
        elif data.required_runs == 0:
            resp = log_and_return_result(
                f"Match already won by {data.batting_team} (Required Runs = 0)",
                f"Congratulations! {data.batting_team} has already won the match!"
            )
            status = 200

        elif data.wickets_left == 0:
            if data.required_runs == 1:
                resp = log_and_return_result(
                    "Match tied! Both teams have equal runs after 20 overs.",
                    "Match tied! Both teams have equal runs!"
                )
            else:
                resp = log_and_return_result(
                    f"Match already won by {data.bowling_team} (Batting team all out)",
                    f"All wickets lost! {data.bowling_team} has won the match!"
                )
            status = 200

        elif data.balls_left == 0:
            if data.required_runs == 1:
                resp = log_and_return_result(
                    "Match tied! Both teams have equal runs after 20 overs.",
                    "Match tied! Both teams have equal runs!"
                )
            else:
                resp = log_and_return_result(
                    f"Overs completed and {data.bowling_team} has won (target not chased).",
                    f"Overs Completed! {data.bowling_team} has won the match!"
                )
            status = 200

        else:
            # ─── Feature engineering ───
            current_score = data.total_runs_x - data.required_runs
            balls_bowled = 120 - data.balls_left
            overs_completed = balls_bowled / 6
            crr = current_score / overs_completed if overs_completed > 0 else 0
            rrr = (data.required_runs * 6) / data.balls_left if data.balls_left > 0 else 0

            logging.info(f"Calculated features: current_score={current_score}, CRR={crr:.2f}, RRR={rrr:.2f}")

            input_df = pd.DataFrame([{
                'batting_team':  data.batting_team,
                'bowling_team':  data.bowling_team,
                'city':          data.city,
                'match_type':    data.match_type,
                'toss_winner':   data.toss_winner,
                'total_runs_x':  data.total_runs_x,
                'required_runs': data.required_runs,
                'balls_left':    data.balls_left,
                'wickets_left':  data.wickets_left,
                'crr':           crr,
                'rrr':           rrr
            }])

            proba = model.predict_proba(input_df)[0][1]
            win_percent = round(proba * 100, 2)

            logging.info(f"Predicted Win Probability: {win_percent}%")
            resp = {"message": f"Win probability of {data.batting_team} is {win_percent}%"}
            status = 200

        # ← use JSONResponse for all JSON bodies
        return JSONResponse(content=resp, status_code=status)

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        ERROR_COUNT.labels(method=method, endpoint=endpoint).inc()
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

    finally:
        # Always record latency & request count
        elapsed = time.time() - start_time
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(elapsed)
        REQUEST_COUNT.labels(
            method=method,
            endpoint=endpoint,
            http_status=str(status)
        ).inc()

#  Uvicorn entrypoint
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
