from flask import Flask,render_template,request
from src.logger import logging
from src.pipelines.prediction_pipeline import PredictPipeline,CustomData

app=Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predictions():
    if request.method == 'GET':
        return render_template('form.html')
    
    else:
        data=CustomData(
        LIMIT_BAL=int(request.form.get('LIMIT_BAL')),
        SEX=int(request.form.get('SEX')),
        EDUCATION=int(request.form.get('EDUCATION')),
        MARRIAGE=int(request.form.get('MARRIAGE')),
        AGE=int(request.form.get('AGE')),
        PAY_0=int(request.form.get('PAY_0')),
        PAY_2=int(request.form.get('PAY_2')),
        PAY_3=int(request.form.get('PAY_3')),
        PAY_4=int(request.form.get('PAY_4')),
        PAY_5=int(request.form.get('PAY_5')),
        PAY_6=int(request.form.get('PAY_6')),
        BILL_AMT1=float(request.form.get('BILL_AMT1')),
        BILL_AMT2=float(request.form.get('BILL_AMT2')),
        BILL_AMT3=float(request.form.get('BILL_AMT3')),
        BILL_AMT4=float(request.form.get('BILL_AMT4')),
        BILL_AMT5=float(request.form.get('BILL_AMT5')),
        BILL_AMT6=float(request.form.get('BILL_AMT6')),
        PAY_AMT1=float(request.form.get('PAY_AMT1')),
        PAY_AMT2=float(request.form.get('PAY_AMT2')),
        PAY_AMT3=float(request.form.get('PAY_AMT3')),
        PAY_AMT4=float(request.form.get('PAY_AMT4')),
        PAY_AMT5=float(request.form.get('PAY_AMT5')),
        PAY_AMT6=float(request.form.get('PAY_AMT6'))
        )
        
        new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        result=predict_pipeline.predict(new_data)
        if result==1:
            results='YES'
        else:
            results='NO'
        return render_template('result.html',final_result=results)
    
    
if __name__ == '__main__':
    app.run(debug=True, port=5001)
