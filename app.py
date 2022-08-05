import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)
# Load the pickled model
model = pickle.load(open('svmmodel_Assignment_6.pkl', 'rb')) 
model_randomforest = pickle.load(open('randomforest_Assignment_6.pkl', 'rb')) 
model_naivebase=pickle.load(open('Assignment_6_Naivebayes.pkl','rb'))
model_KNN = pickle.load(open('KNN_Assignment_6.pkl', 'rb')) 
model_Logestic=pickle.load(open('Logestic_Assignment_6.pkl','rb'))
model_Decision = pickle.load(open('decision_Assignment_6.pkl', 'rb')) 
model_Linear=pickle.load(open('linearregression_Assignment_6.pkl','rb'))


@app.route('/')
def home():
  
    return render_template("index.html")
  
  
@app.route('/pr',methods=['GET'])
def predict():
  exp1 = float(request.args.get('exp1'))
  exp2 = float(request.args.get('exp2'))
  exp3 = float(request.args.get('exp3'))
  exp4 = float(request.args.get('exp4'))
  exp5 = float(request.args.get('exp5'))
  exp6 = float(request.args.get('exp6'))
  exp7 = float(request.args.get('exp7'))
  exp8 = float(request.args.get('exp8'))
  prediction = model.predict([[exp1,exp2,exp3,exp4,exp5,exp6,exp7,exp8]])
  prediction = model_randomforest.predict([[exp1,exp2,exp3,exp4,exp5,exp6,exp7,exp8]])
  prediction = model_naivebase.predict([[exp1,exp2,exp3,exp4,exp5,exp6,exp7,exp8]])
  prediction = model_KNN.predict([[exp1,exp2,exp3,exp4,exp5,exp6,exp7,exp8]])
  prediction = model_Logestic.predict([[exp1,exp2,exp3,exp4,exp5,exp6,exp7,exp8]])
  prediction = model_Decision.predict([[exp1,exp2,exp3,exp4,exp5,exp6,exp7,exp8]])
  prediction = model_Linear.predict([[exp1,exp2,exp3,exp4,exp5,exp6,exp7,exp8]])
  return render_template('index.html', prediction_text='Placements Predicted : {}'.format(prediction))
if __name__ == "__main__":
    app.run(debug=True)
