from flask import Flask, render_template, request
import joblib
model=joblib.load('./model/iris_model.joblib')
types=['setosa', 'versicolor', 'virginica']
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    prediction = model.predict([features])
    species=types[prediction[0]]
    return render_template('index.html', prediction_text=f'The predicted species is {species}')

if __name__ == "__main__":
    app.run(debug=True)
