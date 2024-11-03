from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open('music_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the label encoder to interpret genre predictions
with open('label_encoder.pkl', 'rb') as le_file:
    label_encoder = pickle.load(le_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    age = int(request.form['age'])
    gender = int(request.form['gender'])
    
    # Make a prediction
    prediction = model.predict([[age, gender]])
    genre = label_encoder.inverse_transform(prediction)[0]
    
    return render_template('index.html', recommendation=f'Recommended Genre: {genre}')

if __name__ == '__main__':
    app.run(debug=True)
