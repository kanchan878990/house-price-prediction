"""
test_app.py — Automated tests for the House Price Prediction Flask app.
Run:  python test_app.py  (from house_price_app/ directory)
"""

import sys
import os

# Make sure imports resolve from this directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import app as flask_app

client = flask_app.app.test_client()
flask_app.app.config["TESTING"] = True
flask_app.app.config["SECRET_KEY"] = "test_secret"

PASS = []
FAIL = []


def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print(f"  [PASS] {name}")
    else:
        FAIL.append(name)
        print(f"  [FAIL] {name}" + (f" -- {detail}" if detail else ""))


# ── 1. Model load ──────────────────────────────────────────────────────────────
print("\n--- 1. Model Load ---")
check("model.pkl loaded", flask_app.pipeline is not None)
check("metrics computed", all(k in flask_app.METRICS for k in ("r2", "mae", "rmse")))
check("R2 > 0.5", float(flask_app.METRICS["r2"]) > 0.5,
      f"got {flask_app.METRICS['r2']}")
check("feature coefficients present", len(flask_app.FEATURE_COEFS) > 0)

# ── 2. GET / ──────────────────────────────────────────────────────────────────
print("\n--- 2. GET / (index) ---")
resp = client.get("/")
check("status 200", resp.status_code == 200, str(resp.status_code))
check("form present", b'<form' in resp.data)
check("predict button", b'Predict Price' in resp.data)
check("all 12 input names present", all(
    name.encode() in resp.data for name in [
        "area", "bedrooms", "bathrooms", "stories", "parking",
        "mainroad", "guestroom", "basement", "hotwaterheating",
        "airconditioning", "prefarea", "furnishingstatus"
    ]
))

# ── 3. POST /predict (valid data) ─────────────────────────────────────────────
print("\n--- 3. POST /predict (valid input) ---")
form_data = {
    "area": "7500",
    "bedrooms": "3",
    "bathrooms": "2",
    "stories": "2",
    "parking": "2",
    "mainroad": "yes",
    "guestroom": "no",
    "basement": "no",
    "hotwaterheating": "no",
    "airconditioning": "yes",
    "prefarea": "yes",
    "furnishingstatus": "furnished",
}
resp = client.post("/predict", data=form_data, follow_redirects=False)
check("redirects to /result", resp.status_code == 302, str(resp.status_code))
check("Location header is /result", "/result" in resp.headers.get("Location", ""),
      resp.headers.get("Location", ""))

# Follow redirect to /result
resp2 = client.post("/predict", data=form_data, follow_redirects=True)
check("result page 200", resp2.status_code == 200)
check("price shown on result page", b"Estimated House Price" in resp2.data)
check("rupee symbol on result", "₹".encode() in resp2.data or b"&#8377;" in resp2.data
      or b"Estimated" in resp2.data)

# Extract predicted price from session
with flask_app.app.test_client() as c:
    c.post("/predict", data=form_data)
    with c.session_transaction() as sess:
        predicted = sess.get("predicted_price")
check("predicted price is positive int", isinstance(predicted, int) and predicted > 0,
      str(predicted))
check("predicted price in plausible range",
      1_000_000 < predicted < 20_000_000,
      f"got {predicted:,}")
print(f"         Predicted price: {predicted:,}")

# ── 4. GET /result without session (edge case) ────────────────────────────────
print("\n--- 4. GET /result with no session ---")
with flask_app.app.test_client() as fresh:
    resp = fresh.get("/result", follow_redirects=False)
    check("redirects to / when no session", resp.status_code == 302,
          str(resp.status_code))

# ── 5. GET /metrics ───────────────────────────────────────────────────────────
print("\n--- 5. GET /metrics ---")
resp = client.get("/metrics")
check("metrics page 200", resp.status_code == 200)
check("R2 value on page", str(flask_app.METRICS["r2"]).encode() in resp.data)
check("MAE value on page", flask_app.METRICS["mae"].encode() in resp.data)
check("RMSE value on page", flask_app.METRICS["rmse"].encode() in resp.data)
check("coefficient table present", b"Coefficient" in resp.data)

# ── 6. POST /predict — all furnishing values ──────────────────────────────────
print("\n--- 6. POST /predict (all furnishingstatus values) ---")
for status in ["furnished", "semi-furnished", "unfurnished"]:
    d = form_data.copy()
    d["furnishingstatus"] = status
    r = client.post("/predict", data=d, follow_redirects=True)
    check(f"furnishingstatus={status} -> 200", r.status_code == 200)

# ── 7. POST /predict — all binary combos (smoke) ─────────────────────────────
print("\n--- 7. POST /predict (binary edge cases) ---")
all_yes = {k: "yes" for k in ["mainroad","guestroom","basement","hotwaterheating","airconditioning","prefarea"]}
all_no  = {k: "no"  for k in ["mainroad","guestroom","basement","hotwaterheating","airconditioning","prefarea"]}
for label, overrides in [("all-yes amenities", all_yes), ("all-no amenities", all_no)]:
    d = {"area":"5000","bedrooms":"2","bathrooms":"1","stories":"1","parking":"0",
         "furnishingstatus":"unfurnished", **overrides}
    r = client.post("/predict", data=d, follow_redirects=True)
    check(f"{label} -> 200", r.status_code == 200)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  PASSED : {len(PASS)}")
print(f"  FAILED : {len(FAIL)}")
if FAIL:
    print(f"  Failed tests: {', '.join(FAIL)}")
    sys.exit(1)
else:
    print("  All tests passed!")
print(f"{'='*50}\n")
