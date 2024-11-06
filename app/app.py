import os
import logging
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Set up logging
log_file = os.path.join(os.getcwd(), 'logs', 'app.log')
logging.basicConfig(filename=log_file, level=logging.INFO)

# Load the trained model
model_path = os.path.join(os.getcwd(), 'app', 'model', 'diabetic.pkl')
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form['name']
    email = request.form['email']
    phone = request.form['phone']
    
    pregnancies = float(request.form['pregnancies'])
    glucose = float(request.form['glucose'])
    blood_pressure = float(request.form['blood_pressure'])
    skin_thickness = float(request.form['skin_thickness'])
    insulin = float(request.form['insulin'])
    bmi = float(request.form['bmi'])
    diabetes_pedigree_function = float(request.form['diabetes_pedigree_function'])
    age = float(request.form['age'])
    
    input_data = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]]
    
    # Predict diabetes risk
    prediction = model.predict(input_data)[0]
    result = "High Risk" if prediction == 1 else "Low Risk"
    
    # Log user info and prediction result
    user_data = {
        "name": name,
        "email": email,
        "phone": phone,
        "result": result,
    }
    logging.info(f"User Data: {user_data}")
    
    # You can also save the data to a CSV file if needed
    with open('user_data.csv', 'a') as file:
        file.write(f"{name},{email},{phone},{result}\n")
    
    return render_template('index.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
