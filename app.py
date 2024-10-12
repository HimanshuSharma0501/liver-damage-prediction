from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the optimized Random Forest model and scaler
model = pickle.load(open('rf_model_optimized.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Route to render the HTML form
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle form submission and prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Extract and process user input
    user_input = np.array([[data['N_Days'], data['Age'], data['Bilirubin'], 
                            data['Albumin'], data['Copper'], data['Alk_Phos'], 
                            data['SGOT'], data['Platelets'], data['Prothrombin']]])

    # Convert data to float type
    user_input = user_input.astype(float)
    
    # Scale the user input
    user_input_scaled = scaler.transform(user_input)
    
    # Make prediction
    prediction = model.predict(user_input_scaled)
    
    # Return the prediction result as JSON
    return jsonify({'Predicted Stage': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
