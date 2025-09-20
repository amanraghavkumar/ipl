# ipl-win-probability-predictor
It predicts which team have high chances of winning
Predict the probability of each team winning an Indian Premier League (IPL) match given the ball-by-ball match state.
This repository contains data preprocessing, model training, evaluation, and an interactive dashboard for exploring win probabilities in real time.

Key features

Clean pipeline to convert ball-by-ball match state into model features (score, wickets, overs left, required run rate, momentum features).

Trained gradient-boosted model (XGBoost / LightGBM) for robust probability estimates.

Calibration & evaluation (ROC, Brier score, calibration plots).

Interactive dashboard (Streamlit) to simulate match states and visualize win probability evolution.

Optional API (FastAPI) for programmatic access to the predictor.

Quick demo

Run the Streamlit app:

pip install -r requirements.txt
streamlit run app/streamlit_app.py


Or start the API:

uvicorn app.api:app --reload
# POST match state to /predict with JSON, get win probabilities

Repository structure
.
├─ data/                   # raw and processed datasets (gitignored for large files)
├─ notebooks/              # EDA and model experiments
├─ src/
│  ├─ features.py          # feature engineering and helper functions
│  ├─ preprocess.py        # cleaning & dataset transforms
│  ├─ train.py             # model training script
│  ├─ predict.py           # local prediction helper
│  └─ metrics.py           # evaluation utilities
├─ app/
│  ├─ streamlit_app.py     # interactive UI
│  └─ api.py               # FastAPI endpoints
├─ models/                 # trained model artifacts (gitignored)
├─ requirements.txt
└─ README.md

Input (example match-state JSON)
{
  "inning": 2,
  "batting_team": "Mumbai Indians",
  "bowling_team": "Chennai Super Kings",
  "score": 145,
  "wickets": 5,
  "overs": 15.3,
  "target": 185,
  "last_5_balls_runs": [0,1,4,0,2]
}


Output:

{
  "batting_team_win_prob": 0.34,
  "bowling_team_win_prob": 0.66,
  "model_confidence": 0.92
}

How it works (high level)

Feature engineering: derive overs left, balls remaining, runs required, required run rate, current run rate, wickets in hand, recent momentum (last n balls), venue factor, batting order advantage, etc.

Model: gradient-boosted decision tree (XGBoost or LightGBM) trained on historical ball-by-ball IPL data (e.g., Kaggle IPL dataset or official ball-by-ball logs). Model outputs win probability for the batting side (or either team depending on convention).

Calibration: use isotonic or Platt scaling and compute Brier score to make probability outputs reliable.

Dashboard: visualize probability curve over match, show feature importance, simulate "what-if" match states.

Getting started (local)

Clone the repo:

git clone https://github.com/<your-username>/ipl-win-probability-predictor.git
cd ipl-win-probability-predictor


Install dependencies:

python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt


Prepare data:

Put raw ball-by-ball CSV(s) into data/raw/.

Run preprocessing:

python src/preprocess.py --input data/raw/ipl_balls.csv --out data/processed/train.parquet


Train model:

python src/train.py --train data/processed/train.parquet --out models/xgb_model.pkl


Run app:

streamlit run app/streamlit_app.py
# or start API
uvicorn app.api:app --reload

Example model training arguments

src/train.py supports:

choice of model: --model xgboost|lightgbm

cross-validation folds: --cv 5

calibration toggle: --calibrate True

Evaluation & metrics

Use Brier score for probability quality.

Use ROC AUC for discrimination.

Plot calibration curve and reliability diagrams.

Backtest time-slices (e.g., evaluate model on overs 1–10, 11–20 separately to check stability).

Tips for better performance

Add match-level context (venue, toss winner, team strength, home/away).

Use recent season weighting — teams evolve year-to-year.

Feature interactions (e.g., runs_required × wickets_in_hand) often boost performance.

Consider ensembling XGBoost + LightGBM for more robust outputs.

Data sources

Public ball-by-ball IPL datasets (e.g., Kaggle).

Official IPL match logs (if accessible).
(Remember to respect dataset licenses and do not commit raw large datasets to the repo; add them to .gitignore.)

API contract (example)

POST /predict
Body: JSON match-state (see example above)
Response: { batting_team_win_prob, bowling_team_win_prob, meta }

Contributing

Add new features in src/features.py.

Keep notebooks for experiments in notebooks/ and summarize in docs/.

Add unit tests for preprocessing and prediction logic.

Open an issue or PR — happy to review!
