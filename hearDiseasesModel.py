import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

# Load the data
data = pd.read_csv("/content/heart.csv")

# Split the data
x = data.drop(columns='target', axis=1)
y = data['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

# Define the transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), x.columns),
        ('cat', OneHotEncoder(), ['sex', 'fbs', 'exang'])
    ])

# Create a pipeline with the preprocessor and the model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# Train the logistic regression model
model.fit(x_train, y_train)

# Predict on the training set
y_train_pred = model.predict(x_train)

# Calculate accuracy on the training set
accuracy = accuracy_score(y_train_pred, y_train)
print(f"Accuracy on the training set: {accuracy}")

# Save the model
# joblib.dump(model, 'heart_disease_model.pkl')
