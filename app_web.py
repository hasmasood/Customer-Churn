
from flask import Flask, request, app, jsonify, url_for, render_template
import pickle
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
import numpy as np

app = Flask('customerchurn')
model, scaler, dv = pickle.load( open('model_stack_C=1.0.pkl', 'rb') )

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_web', methods=['POST'])
def predict():

    keys = [x for x in request.form.keys()]
    values = [x for x in request.form.values()]
    
    #Some numbers are converted to str, so we need to fix that.
    for i in range(0,len(values)):
        try:
            values[i] = float(values[i])
            print('changed')
        except ValueError:
            values[i] = str(values[i])
            print(values[i])

    data_as_dic = [dict(zip(keys, values))] #Our original sample is list of dictionaries so I used []
    print('Data type:', type(data_as_dic))
    print('Data shape:', len(data_as_dic[0]))
    Xohe_as_array = dv.transform(data_as_dic)
    print('Data type:', type(Xohe_as_array))
    print('Data shape:', Xohe_as_array.shape)
    output = model.predict_proba(scaler.transform(Xohe_as_array))[:,1]
    print('Predicted value:', '%.2f' % (output[0]))
    return render_template('home.html', prediction_text= "The churn probability is {}".format(output[0]))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
