<<<<<<< HEAD
# Maritime Predictive Maintenance (PdM) — ML Application

A thesis-support project: an end-to-end machine learning pipeline that predicts
when a ship's engine/propulsion system is likely to need maintenance, based on
simulated sensor telemetry (exhaust temperature, coolant temperature, lube-oil
pressure, vibration, RPM, fuel rate, oil particle count, hull speed).

## Project structure

```
maritime_pdm/
├── data/
│   └── maritime_sensor_data.csv     # synthetic fleet sensor dataset (generated)
├── src/
│   ├── generate_data.py             # synthetic data generator
│   ├── features.py                  # rolling-window feature engineering
│   ├── train.py                     # trains & compares 3 classifiers, saves best
│   ├── predict.py                   # CLI batch inference
│   └── app.py                       # Streamlit dashboard
├── models/
│   └── best_model.joblib            # trained model bundle
├── figures/                         # confusion matrix, ROC curve, feature importance
├── outputs/
│   └── metrics.json                 # evaluation metrics
├── requirements.txt
└── README.md
```

## Quick start

```bash
pip install -r requirements.txt

# 1. Generate the synthetic fleet dataset
python src/generate_data.py

# 2. Train and evaluate models (Logistic Regression, Random Forest, Gradient Boosting)
python src/train.py

# 3a. Batch predictions from the command line
python src/predict.py --input data/maritime_sensor_data.csv --output risk_scores.csv

# 3b. Or launch the interactive dashboard
streamlit run src/app.py
```

## How it works

1. **Data**: Each simulated engine "unit" runs for a random lifetime between
   120–380 operating cycles. Early cycles look like healthy baseline readings;
   as a unit approaches failure, sensor values drift following a nonlinear
   degradation curve — mirroring how real machinery degrades (slowly at first,
   then rapidly).
2. **Labeling**: A cycle is labeled `1` (at-risk) if it falls within 30 cycles
   of that unit's failure point, and `0` otherwise. This turns the problem into
   binary classification, which is easier to act on operationally than a raw
   remaining-useful-life regression, though `RUL` is also included in the data
   for anyone who wants to extend the project to regression.
3. **Features**: In addition to raw sensor values, rolling mean, rolling
   standard deviation, and rolling trend (slope) are computed over a 10-cycle
   window per sensor — these trend features are what let the model detect
   *drift*, not just anomalous single readings.
4. **Train/test split**: Split by `unit_id` (`GroupShuffleSplit`), so the model
   is evaluated on engines it has never seen — this avoids the common mistake
   of leaking cycles from the same unit into both train and test sets.
5. **Models**: Logistic Regression (baseline), Random Forest, and Gradient
   Boosting are trained and compared on precision, recall, F1, and ROC-AUC on
   the held-out units; the best model by ROC-AUC is saved.
6. **Deployment**: `predict.py` gives batch CSV-in/CSV-out inference for
   integration into a maintenance workflow; `app.py` is a Streamlit dashboard
   for engineers to explore risk scores interactively.

## Notes for extending this into a full thesis project

- Replace `generate_data.py` with a real telemetry feed (e.g. NMEA 2000 engine
  gateway data, or a public dataset such as NASA C-MAPSS turbofan degradation,
  which this generator is structurally inspired by) once real fleet data is
  available.
- Add a regression head to predict Remaining Useful Life (`RUL` column is
  already present) alongside the classification alert.
- Add model explainability (e.g. SHAP values) for regulator- and crew-facing
  transparency.
- Containerize `app.py` with Docker and expose `predict.py` as a REST endpoint
  (FastAPI) for integration with a ship's onboard monitoring system.
=======
# Maritime-Predictive-Maintenance-App
My thesis for the University of Pireaus, school of Digital Systems, which consists of a machine learning application for maritime predictive maintenance.
>>>>>>> 377d5f33d5f6c550e69c5970cd68299b7e9f78b0
