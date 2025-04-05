from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scaler
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form values
        features = [
            float(request.form['Age']),
            float(request.form['Income']),
            float(request.form['LoanAmount']),
            float(request.form['CreditScore']),
            float(request.form['MonthsEmployed']),
            float(request.form['NumCreditLines']),
            float(request.form['InterestRate']),
            float(request.form['LoanTerm']),
            float(request.form['DTIRatio']),
            int(request.form['HasCoSigner']),
            int(request.form['HasMortgage']),
            int(request.form['HasDependents'])
        ]

        # Scale and predict
        scaled = scaler.transform([features])
        prediction = model.predict(scaled)[0]

        result = "High risk of Default ❌" if prediction == 1 else "Low risk of Default ✅"
        return render_template('index.html', prediction_text=f'Prediction: {result}')

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
