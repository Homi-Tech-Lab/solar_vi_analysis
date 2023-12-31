import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import load_model
from flask import Flask, render_template, request, jsonify

# Load and preprocess data
def load_and_preprocess_data():
    dataset = pd.read_csv('../Solar_categorical.csv')
    X = dataset.iloc[:3000, 0:7].values
    y = dataset.iloc[:3000, 7].values

    encoder = LabelEncoder()
    X[:, 6] = encoder.fit_transform(X[:, 6])
    y = encoder.fit_transform(y)
    y = to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler, encoder

# Initialize model and preprocessors
X_train, X_test, y_train, y_test, scaler, encoder = load_and_preprocess_data()
new_model = load_model('../fault_model.model')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sensor_values = [float(request.form['sensor1']), float(request.form['sensor2']),
                      float(request.form['sensor3']), float(request.form['sensor4']),
                      float(request.form['irradiance']), float(request.form['temperature']),
                      int(request.form['sunny'])]

    input_data = scaler.transform(np.array([sensor_values]))
    predictions = new_model.predict(input_data)
    original_label = encoder.inverse_transform([np.argmax(predictions)])[0]

    return jsonify({'result': original_label})

if __name__ == '__main__':
    app.run(debug=True)
