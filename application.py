from flask import Flask,request,jsonify,render_template

import pickle
import numpy as np
import pandas as pd
import os
import pickle
from huggingface_hub import hf_hub_download

rf_regressor = None
preprocessor = None

application = Flask(__name__)
app = application

def load_models():
    global rf_regressor, preprocessor

    if rf_regressor is None:
        model_path = hf_hub_download(
            repo_id="Annabeth08/crop-yield-model",
            filename="rf_regressor.pkl"
        )

rf_regressor=pickle.load(open('models/rf_regressor.pkl','rb'))
preprocessor=pickle.load(open('models/preprocessor.pkl','rb'))
application=Flask(__name__)   

app=application  

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_data', methods=['GET','POST'])
def predict():

    if request.method == 'POST':

        load_models()
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

        transformed_data = preprocessor.transform(input_df)
        result = rf_regressor.predict(transformed_data)

        return jsonify({
            "prediction": int(result[0]),
            "crop": Item,
            "region": Area
        })

    else:
        return render_template('home.html')

if __name__=="__main__":  
    app.run(debug=True)     

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)