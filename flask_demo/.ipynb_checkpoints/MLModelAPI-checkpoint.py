import pickle
from flask import Flask,request
from flasgger import Swagger
import numpy as np
import pandas as pd

with open('.\PickleFiles\RandomForest.pkl','rb') as model_file:
       model=pickle.load(model_file)
	   
	   
app = Flask(__name__)
swagger=Swagger(app)
@app.route('/predict')
def predict_iris():
    """Example endpoint returing a prediction
    ---
    parameters:
            - name: s_l1
              in: query
              type: number
              required: true
            - name: s_w1
              in: query
              type: number
              required: true
            - name: p_l1
              in: query
              type: number
              required: true
            - name: p_w1
              in: query
              type: number
              required: true
    """
    s_l1=request.args.get('s_l1')
    s_w1=request.args.get('s_w1')
    p_l1=request.args.get('p_l1')
    p_w1=request.args.get('p_w1')
    prediction=model.predict(np.array([[s_l1,s_w1,p_l1,p_w1]]))
    return str(prediction)


@app.route('/predict_file',methods=['POST'])
def predict_iris_file():
    """Example endpoint returing a prediction for a fiile input
    ---
        parameters:
            - name: input_file
              in: formData
              type: file
              required: true
      """        
    input_data=pd.read_csv(request.files.get('input_file'),header=None)
    prediction=model.predict(input_data)
    return str(list(prediction))  

if __name__ == '__main__':
    app.run()             	   