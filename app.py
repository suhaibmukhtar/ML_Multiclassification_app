import pandas as pd
from flask import Flask,render_template,request
import pickle
import sklearn
import numpy as np
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
#importing model
model=pickle.load(open('sentiment.pkl','rb')) #read binary
app=Flask(__name__)

@app.route("/") #where to find html webpage
def index():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict_class():
    # Retrieve form data and handle empty values
    data = []
    for i in range(1, 19):
        field_name = str(i)
        #retrives the data from each field of the form
        field_value = request.form.get(field_name)
        if field_value:
            data.append(int(field_value))
        else:
            # Handle empty values here, for example, you can replace them with 0
            data.append(0)

    # Convert data to numpy array and reshape it for prediction
    data_array = np.array(data).reshape(1, -1)

    # Predict using the model
    result = model.predict(data_array)

    return render_template('index.html',result=result[0]) #to show result also on same page
if __name__=='__main__':
    app.run(host='0.0.0.0',port=8081) #for hosting
#
