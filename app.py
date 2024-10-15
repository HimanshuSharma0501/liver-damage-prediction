from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the saved Random Forest model
with open('rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load encoders and scaler
with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define the home route that renders the HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Define the route to predict liver damage
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        input_features = [
            int(request.form['N_Days']),
            request.form['Status'],
            request.form['Drug'],
            int(request.form['Age']),
            request.form['Sex'],
            request.form['Ascites'],
            request.form['Hepatomegaly'],
            request.form['Spiders'],
            request.form['Edema'],
            float(request.form['Bilirubin']),
            float(request.form['Cholesterol']),
            float(request.form['Albumin']),
            float(request.form['Copper']),
            float(request.form['Alk_Phos']),
            float(request.form['SGOT']),
            float(request.form['Tryglicerides']),
            float(request.form['Platelets']),
            float(request.form['Prothrombin']),
        ]

        # Convert categorical features to numeric values using encoders
        input_features[1] = encoders['Status'].transform([input_features[1]])[0]
        input_features[2] = encoders['Drug'].transform([input_features[2]])[0]
        input_features[4] = encoders['Sex'].transform([input_features[4]])[0]
        input_features[5] = encoders['Ascites'].transform([input_features[5]])[0]
        input_features[6] = encoders['Hepatomegaly'].transform([input_features[6]])[0]
        input_features[7] = encoders['Spiders'].transform([input_features[7]])[0]
        input_features[8] = encoders['Edema'].transform([input_features[8]])[0]

        # Convert the input to a NumPy array and scale it
        input_data = np.array(input_features).reshape(1, -1)
        input_data_scaled = scaler.transform(input_data)

        # Predict the stage using the model
        prediction = model.predict(input_data_scaled)
        
        # Mapping of prediction result
        stage_map = {0: 'Stage 0', 1: 'Stage 1', 2: 'Stage 2', 3: 'Stage 3'}
        predicted_stage = stage_map[prediction[0]]

        # Return the prediction result
        return render_template('result.html', prediction=predicted_stage)

if __name__ == '__main__':
    app.run(debug=True)
