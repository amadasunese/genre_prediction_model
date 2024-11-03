from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model
with open('music_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

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
    genre = prediction[0]  # Directly use the prediction result as the genre
    
    return render_template('index.html', recommendation=f'Recommended Genre: {genre}')




if __name__ == '__main__':
    app.run(debug=True)
