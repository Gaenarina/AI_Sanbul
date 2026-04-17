import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)
PORT = int(os.environ.get("PORT", 5000))

print("loading pipeline...")
pipeline = joblib.load("preprocess_pipeline.pkl")
print("pipeline loaded")

print("loading numpy weights...")
weights = np.load("mlp_weights.npz")
W1, b1 = weights["W1"], weights["b1"]
W2, b2 = weights["W2"], weights["b2"]
W3, b3 = weights["W3"], weights["b3"]
W4, b4 = weights["W4"], weights["b4"]
print("weights loaded")


def relu(x):
    return np.maximum(0, x)


def mlp_predict(x):
    z1 = relu(np.dot(x, W1) + b1)
    z2 = relu(np.dot(z1, W2) + b2)
    z3 = relu(np.dot(z2, W3) + b3)
    y = np.dot(z3, W4) + b4
    return y


@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html")


@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
        try:
            longitude = float(request.form["longitude"])
            latitude = float(request.form["latitude"])
            month = request.form["month"]
            day = request.form["day"]
            avg_temp = float(request.form["avg_temp"])
            max_temp = float(request.form["max_temp"])
            max_wind_speed = float(request.form["max_wind_speed"])
            avg_wind = float(request.form["avg_wind"])

            input_df = pd.DataFrame([{
                "longitude": longitude,
                "latitude": latitude,
                "month": month,
                "day": day,
                "avg_temp": avg_temp,
                "max_temp": max_temp,
                "max_wind_speed": max_wind_speed,
                "avg_wind": avg_wind
            }])

            transformed = pipeline.transform(input_df)

            if hasattr(transformed, "toarray"):
                transformed = transformed.toarray()

            pred_log = mlp_predict(transformed)[0][0]
            pred_area = np.exp(pred_log) - 1

            if pred_area < 0:
                pred_area = 0.0

            return render_template(
                "result.html",
                pred_area=round(float(pred_area), 2)
            )

        except Exception as e:
            return render_template("result.html", error=str(e))

    return render_template("prediction.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
