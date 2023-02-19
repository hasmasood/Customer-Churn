
from flask import Flask, request,jsonify
import pickle
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
import numpy as np

app = Flask('customerchurn')
model, scaler, dv = pickle.load( open('model_stack_C=1.0.pkl', 'rb') )

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data_as_dic=request.get_json()
    print('Data type:', type(data_as_dic))
    print('Data shape:', len(data_as_dic[0]))
    Xohe_as_array = dv.transform(data_as_dic)
    print('Data type:', type(Xohe_as_array))
    print('Data shape:', Xohe_as_array.shape)
    output = model.predict_proba(scaler.transform(Xohe_as_array))[:,1]
    print('Predicted value:', '%.2f' % (output[0]))
    return jsonify(output[0])

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
