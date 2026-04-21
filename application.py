from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
import os

# Flask app
application = Flask(__name__)
app = application

# ✅ Load models safely
base_dir = os.path.dirname(__file__)

preprocessor_path = os.path.join(base_dir, "preprocessor.pkl")
rf_model_path = os.path.join(base_dir, "rf_regressor.pkl")

preprocessor = pickle.load(open(preprocessor_path, "rb"))
rf_regressor = pickle.load(open(rf_model_path, "rb"))

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict_data', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        Area = data['area']
        Item = data['item']
        Year = int(data['year'])
        rainfall = float(data['rainfall'])
        pesticides = float(data['pesticides'])
        avg_temp = float(data['avg_temp'])

        input_df = pd.DataFrame([{
            'Area': Area,
            'Item': Item,
            'Year': Year,
            'average_rain_fall_mm_per_year': rainfall,
            'pesticides_tonnes': pesticides,
            'avg_temp': avg_temp
        }])

        # Transform + Predict
        transformed_data = preprocessor.transform(input_df)
        result = rf_regressor.predict(transformed_data)

        return jsonify({
            "prediction": int(result[0]),
            "crop": Item,
            "region": Area
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# Run app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
