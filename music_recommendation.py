# Import pandas and other necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# Load and explore the dataser

music_df = pd.read_csv('music.csv')

# Display the first few rows of the dataset
print(f'First few rows of the dataset')
music_df.head()


# Summary information of the dataset
print('summary information of the dataset')
music_df.info()

# Data Cleaning

# Check for missing values and duplicates
missing_values = music_df.isnull().sum()
duplicates = music_df.duplicated().sum()

print(f'Missing Values: {missing_values, duplicates} ')

# Data Preparation

# Define features (X) and label (y)
X = music_df[['age', 'gender']]
y = music_df['genre']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Show the shapes of the training and testing sets
print(f'Training Set Shape: {X_train.shape, X_test.shape, y_train.shape, y_test.shape} ')

# Model Building

# Initialize the Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)

# Train the model on the training data
dt_model.fit(X_train, y_train)


# Make Predictions

# Make predictions on the test set
y_pred = dt_model.predict(X_test)

# Display predictions
print("Predictions:", y_pred)


# Model Evaluation

# Calculate and print the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')
