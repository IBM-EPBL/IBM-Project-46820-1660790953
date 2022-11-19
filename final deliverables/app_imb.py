import pickle
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas
import requests
from flask import Flask, render_template, request
API_KEY = "qJSD5ROm29i2iJGncQiHjCmDMgTx_563xkzVMRZ3Wvw3"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":
 API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

app = Flask(__name__)
model=pickle.load(open('heart.pkl','rb'))
scale=pickle.load(open('scale.pkl','rb'))
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=["POST","GET"])
def predict():
    input_feature=[x for x in request.form.values()]
    feature_values=[np.array(input_feature)]
    names=[['age','sex','cp','testbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']]
    response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/cae560db-c5dc-4eba-b799-99e910c25da3/predictions?version=2022-11-04', json=names,
    headers={'Authorization': 'Bearer ' + mltoken})
    print("Scoring response")
    print(response_scoring.json())
    data=pandas.DataFrame(feature_values,columns=names)
    data=scale.fit_transform(data)
    data=pandas.DataFrame(data,columns=names)
    prediction =model.predict(data)
    pred_prob=model.predict_proba(data)
    print(prediction)
    if prediction == "Yes":
        return render_template("chance.html")
    else:
        return render_template("nochance.html")
if __name__ == "__main__": 
  app.run(debug=True)            
