# House Price Prediction

A full-stack web application that predicts house prices using a **Linear Regression** model trained on the Housing dataset.

Built with **Python**, **Flask**, **scikit-learn**, and **Bootstrap 5**.

---

## Project Structure

```
house_price_app/
├── app.py                  # Flask application
├── requirements.txt        # Python dependencies
├── model/
│   ├── train.py            # Training script
│   └── model.pkl           # Saved pipeline (generated)
├── templates/
│   ├── base.html           # Shared layout
│   ├── index.html          # Prediction form
│   ├── result.html         # Result page
│   └── metrics.html        # Model performance page
└── static/
    └── style.css           # Custom CSS
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model (first time only)

Run from the project root (where `Housing.csv` lives):

```bash
python house_price_app/model/train.py
```

This creates `house_price_app/model/model.pkl`.

Expected output:
```
R²   : 0.6529
MAE  : 970,043
RMSE : 1,324,507

Model saved -> ...model.pkl
```

### 3. Start the web app

```bash
cd house_price_app
python app.py
```

Open your browser at: **http://127.0.0.1:5000**

---

## Pages

| URL | Description |
|---|---|
| `/` | Prediction form — enter house details |
| `/predict` | POST endpoint — runs inference, redirects to result |
| `/result` | Displays the predicted price + input summary |
| `/metrics` | Model R², MAE, RMSE and feature coefficient table |

---

## Dataset

`Housing.csv` — 545 rows, 13 columns.

| Feature | Type |
|---|---|
| area | Numeric |
| bedrooms, bathrooms, stories, parking | Numeric |
| mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea | Binary (yes/no) |
| furnishingstatus | Categorical (furnished / semi-furnished / unfurnished) |
| **price** | **Target** |

---

## Model Details

- **Algorithm**: Linear Regression (`sklearn.linear_model.LinearRegression`)
- **Preprocessing**: Binary columns mapped to 0/1; `furnishingstatus` one-hot encoded with `drop="first"`
- **Pipeline**: `ColumnTransformer` → `LinearRegression` (saved as a single `model.pkl`)
- **Train/Test Split**: 80% / 20%, `random_state=42`

---

## Future Improvements (To-Do)

- [ ] Add more models: Ridge, Lasso, Random Forest, Gradient Boosting
- [ ] Add cross-validation for more robust evaluation
- [ ] Add feature engineering (e.g., price-per-sqft, room ratio)
- [ ] Add EDA charts (price distribution, correlation heatmap) on the metrics page
- [ ] Add input validation and friendly error messages on the form
- [ ] Add a comparison page to predict multiple houses side-by-side
- [ ] Deploy to a cloud platform (Heroku / Render / Railway)
- [ ] Add authentication if deployed publicly
- [ ] Export prediction history to CSV
- [ ] Add a REST API endpoint (`/api/predict`) for programmatic access

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Web Framework | Flask |
| ML Library | scikit-learn |
| Data Handling | pandas, numpy |
| Model Persistence | joblib |
| Frontend | Jinja2, Bootstrap 5 |
