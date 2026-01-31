from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained model
with open("linear_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        input_df = pd.DataFrame([{
            "age": float(data["age"]),
            "bmi": float(data["bmi"]),
            "children": int(data["children"]),   # ✅ NO strip here
            "sex": str(data["sex"]).strip().lower(),
            "smoker": str(data["smoker"]).strip().lower(),
            "region": str(data["region"]).strip().lower()
        }])

        prediction = model.predict(input_df)

        return jsonify({
            "prediction": round(float(prediction[0]), 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
