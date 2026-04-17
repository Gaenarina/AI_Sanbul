import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from flask import Flask, render_template, request

app = Flask(__name__)
PORT = int(os.environ.get("PORT", 5000))

print("loading model...")
model = tf.keras.models.load_model("fires_model.keras")
print("model loaded")

print("loading pipeline...")
pipeline = joblib.load("preprocess_pipeline.pkl")
print("pipeline loaded")


@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html")


@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
        try:
            print("POST /prediction start")

            longitude = float(request.form["longitude"])
            latitude = float(request.form["latitude"])
            month = request.form["month"]
            day = request.form["day"]
            avg_temp = float(request.form["avg_temp"])
            max_temp = float(request.form["max_temp"])
            max_wind_speed = float(request.form["max_wind_speed"])
            avg_wind = float(request.form["avg_wind"])

            print("form parsed")

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

            print("dataframe created")

            transformed = pipeline.transform(input_df)
            print("pipeline transformed")

            if hasattr(transformed, "toarray"):
                transformed = transformed.toarray()
                print("converted to dense")

            pred_log = model.predict(transformed, verbose=0)[0][0]
            print("prediction done")

            pred_area = np.exp(pred_log) - 1

            if pred_area < 0:
                pred_area = 0.0

            return render_template(
                "result.html",
                pred_area=round(float(pred_area), 2)
            )

        except Exception as e:
            print("ERROR:", str(e))
            return render_template("result.html", error=str(e))

    return render_template("prediction.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
