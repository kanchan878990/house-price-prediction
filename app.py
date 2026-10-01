"""
app.py — House Price Prediction Flask Application
Run:  python app.py
"""

import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ── App setup ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "housepriceapp_secret_2024"

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
DATA_PATH  = os.path.join(BASE_DIR, "..", "Housing.csv")

# ── Load model & compute metrics once at startup ───────────────────────────────
pipeline = joblib.load(MODEL_PATH)

# Reload dataset to compute metrics + feature importances
df = pd.read_csv(DATA_PATH)
BINARY_COLS = ["mainroad", "guestroom", "basement",
               "hotwaterheating", "airconditioning", "prefarea"]

_df = df.copy()
for col in BINARY_COLS:
    _df[col] = _df[col].map({"yes": 1, "no": 0})

X_all = _df.drop(columns=["price"])
y_all = _df["price"]
_, X_test, _, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

y_pred  = pipeline.predict(X_test)
METRICS = {
    "r2":   round(r2_score(y_test, y_pred), 4),
    "mae":  f"{mean_absolute_error(y_test, y_pred):,.0f}",
    "rmse": f"{np.sqrt(mean_squared_error(y_test, y_pred)):,.0f}",
}

# Build feature-name → coefficient table
ohe          = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
ohe_features = list(ohe.get_feature_names_out(["furnishingstatus"]))
feature_names = (
    ["area", "bedrooms", "bathrooms", "stories", "parking"] +
    BINARY_COLS +
    ohe_features
)
coefficients = list(pipeline.named_steps["regressor"].coef_)
FEATURE_COEFS = sorted(
    zip(feature_names, [round(c, 2) for c in coefficients]),
    key=lambda x: abs(x[1]),
    reverse=True,
)


# ── Helper ─────────────────────────────────────────────────────────────────────
def _build_input_df(form):
    """Convert raw form POST data into a DataFrame the pipeline can predict on."""
    binary_map = {"yes": 1, "no": 0}
    data = {
        "area":             [int(form["area"])],
        "bedrooms":         [int(form["bedrooms"])],
        "bathrooms":        [int(form["bathrooms"])],
        "stories":          [int(form["stories"])],
        "parking":          [int(form["parking"])],
        "mainroad":         [binary_map[form["mainroad"]]],
        "guestroom":        [binary_map[form["guestroom"]]],
        "basement":         [binary_map[form["basement"]]],
        "hotwaterheating":  [binary_map[form["hotwaterheating"]]],
        "airconditioning":  [binary_map[form["airconditioning"]]],
        "prefarea":         [binary_map[form["prefarea"]]],
        "furnishingstatus": [form["furnishingstatus"]],
    }
    return pd.DataFrame(data)


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    input_df        = _build_input_df(request.form)
    predicted_price = round(float(pipeline.predict(input_df)[0]))
    session["predicted_price"] = predicted_price
    session["input_data"]      = request.form.to_dict()
    return redirect(url_for("result"))


@app.route("/result")
def result():
    predicted_price = session.get("predicted_price")
    input_data      = session.get("input_data", {})
    if predicted_price is None:
        return redirect(url_for("index"))
    formatted_price = f"{predicted_price:,}"
    return render_template("result.html",
                           predicted_price=formatted_price,
                           input_data=input_data)


@app.route("/metrics")
def metrics():
    return render_template("metrics.html",
                           metrics=METRICS,
                           feature_coefs=FEATURE_COEFS)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
