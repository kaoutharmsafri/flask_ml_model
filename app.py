from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
#app=Flask(__name__)
app=Flask(__name__, template_folder='templates', static_folder='static')
#@app.route('/test')
#def test():
#    return "Flask is being used by kokie"

#load model prediction to imrove our app   

mul_reg=open("multiple_linear_model.pkl","rb")
ml_model=joblib.load(mul_reg)

@app.route('/')
def home():
    return render_template('home.html')

@app.route("/predict",methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        try:
            selected_state = request.form['state']
            if selected_state == 'NewYork':
                NewYork = 1.0
                California = 0.0
                Florida = 0.0
            elif selected_state == 'California':
                NewYork = 0.0
                California = 1.0
                Florida = 0.0
            elif selected_state == 'Florida':
                NewYork = 0.0
                California = 0.0
                Florida = 1.0
            else:
                raise ValueError("Please select a state")
            
            RnD_Spend = float(request.form['RnD_Spend'])
            Admin_Spend = float(request.form['Admin_Spend'])
            Market_Spend = float(request.form['Market_Spend'])
            pred_args=[NewYork,California,Florida,RnD_Spend,Admin_Spend,Market_Spend]
            pred_args_arr=np.array(pred_args)
            pred_args_arr=pred_args_arr.reshape(1,-1)
            model_prediction=ml_model.predict(pred_args_arr)
            model_prediction=round(float(model_prediction),2)
        except ValueError:
            return "Please check if the values are entered correctly"  
        return render_template('predict.html',prediction=model_prediction)
if __name__ == '__main__':
    app.run(host='0.0.0.0')