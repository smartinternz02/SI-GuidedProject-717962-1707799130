from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the machine learning model
model = joblib.load("floods.save")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    temp = float(request.form['temp'])
    humidity = float(request.form['humidity'])
    cloud_cover = float(request.form['cloud_cover'])
    annual = float(request.form['annual'])
    jan_feb = float(request.form['jan_feb'])
    mar_may = float(request.form['mar_may'])
    jun_sep = float(request.form['jun_sep'])
    oct_dec = float(request.form['oct_dec'])
    avgjune = float(request.form['avgjune'])
    sub = float(request.form['sub'])
    
    # Make a prediction using the machine learning model
    input_data = np.array([[temp, humidity, cloud_cover, annual, jan_feb, mar_may, jun_sep, oct_dec,avgjune, sub]])
    prediction = model.predict(input_data)

    # Display the appropriate output page based on the prediction result
    if prediction[0] == 1:
        return render_template('flood.html')
    else:
        return render_template('noflood.html')

if __name__ == '__main__':
    app.run(debug=True)
