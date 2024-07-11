from flask import Flask, render_template, request
import joblib

model = joblib.load('./model/trained_model.joblib')
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    result = None
    re = None
    if request.method == 'POST':
        sex = request.form['sex']
        pclass = request.form['pclass']
        
        result = model.predict([[pclass,sex]])
        if result[0]==0:
            re='Not Survived'
        else:
            re='Survived'
    return render_template('prediction.html', result=re)

if __name__ == '__main__':
    app.run(debug=True)
