from flask import Flask, render_template, request
import numpy as np
import pickle
import joblib
app = Flask(__name__)
filename = 'week4.pkl'
#model = pickle.load(open(filename, 'rb'))
model = joblib.load(filename)
#model = joblib.load(filename)
@app.route('/')
def index(): 
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    Aspect_Ratio = request.form['Aspect_Ratio']
    Uniformity = request.form['Uniformity']
    Elongation = request.form['Elongation']
    Solidity = request.form['Solidity']

    
      
    pred = model.predict(np.array([[Aspect_Ratio, Uniformity, Elongation, Solidity ]]))
    print(pred)
    return render_template('index.html', predict=str(pred))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)