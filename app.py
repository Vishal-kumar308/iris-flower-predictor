from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    inputs = [float(request.form[i]) for i in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    prediction = model.predict([inputs])[0]
    return f'<h3>Predicted Iris Species: {prediction}</h3>'

if __name__ == '__main__':
    app.run()
