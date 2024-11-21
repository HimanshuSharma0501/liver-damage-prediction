from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = pickle.load(open('rf_modelImprovised.pkl', 'rb'))
scaler = pickle.load(open('scalerImprovised.pkl', 'rb'))

status_mapping = {'C': 0, 'CL': 1, 'D': 2}  
drug_mapping = {'Placebo': 0, 'D-penicillamine': 1}
sex_mapping = {'M': 1, 'F': 0}
yes_no_mapping = {'Y': 1, 'N': 0}
edema_mapping = {'N': 0, 'S': 1, 'Y': 2}  

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input features from the form
    N_Days = int(request.form['N_Days'])
    Status = status_mapping.get(request.form['Status'], 0) 
    Drug = drug_mapping.get(request.form['Drug'], 0)  
    Age = float(request.form['Age'])
    Sex = sex_mapping.get(request.form['Sex'], 0)  
    Ascites = yes_no_mapping.get(request.form['Ascites'], 0)
    Hepatomegaly = yes_no_mapping.get(request.form['Hepatomegaly'], 0)
    Spiders = yes_no_mapping.get(request.form['Spiders'], 0)
    Edema = edema_mapping.get(request.form['Edema'], 0)
    Bilirubin = float(request.form['Bilirubin'])
    Cholesterol = float(request.form['Cholesterol'])
    Albumin = float(request.form['Albumin'])
    Copper = float(request.form['Copper'])
    Alk_Phos = float(request.form['Alk_Phos'])
    SGOT = float(request.form['SGOT'])
    Tryglicerides = float(request.form['Tryglicerides'])
    Platelets = float(request.form['Platelets'])
    Prothrombin = float(request.form['Prothrombin'])

    
    input_features = np.array([[
        N_Days, Status, Drug,Age, Sex, Ascites, Hepatomegaly, Spiders, Edema, Bilirubin, Cholesterol, 
        Albumin, Copper, Alk_Phos, SGOT, Tryglicerides, Platelets, Prothrombin
    ]])

    input_features_scaled = scaler.transform(input_features)

    
    prediction = model.predict(input_features_scaled)

    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
