# Music Recommendation System

This project is a simple Music Recommendation System that suggests a music genre based on user demographics like age and gender. The model was built using a Decision Tree Classifier.

## Dataset
The dataset (music.csv) contains information about:
1. Age: The listener's age
2. Gender: The listener's gender (encoded as 1 for male, 0 for female)
3. Genre: The preferred genre of music

## Project Steps
Steps to Run the Project
1. Load and Inspect Data: Load the dataset and inspect the columns to understand the data structure.
2. Data Cleaning: Check for missing values and duplicates. If any are found, they should be removed or handled appropriately.
3. Data Preparation: Encode categorical variables like genre to numeric labels. Split the data into features (age, gender) and labels (genre). Split the dataset into training and testing sets.
4. Model Building: A Decision Tree Classifier is used to train the model on the training set.
5. Make Predictions: Use the trained model to make predictions on the test set.
6. Model Evaluation: Evaluate the model's accuracy using accuracy metrics.
7. Save the trained model for deployment.

## Requirements
The project requires Python and the following libraries:
1. pandas
2. sklearn
3. Jupyter Notebook (if you want to run the script in a Jupyter Notebook)


## Future Improvements
The dataset used to train the model for this application has only 18 entries. For improve recommendation accuracy, add more demographic features. Also, training the model with a larger dataset ensure better accuracy.

## AUTHOR

Ese Amadasun
amadasunese@gmail.com
