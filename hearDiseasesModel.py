import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
import joblib

data = pd.read_csv("/content/heart.csv")

# Assuming 'sex' is the only categorical feature
categorical_features = ['sex']
numeric_features = [col for col in data.columns if col not in ['target'] + categorical_features]

# Create a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# Create a pipeline with the preprocessor and the model
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', LogisticRegression())])

# Split the data
x = data.drop(columns='target', axis=1)
y = data['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

# # Train the model
# model.fit(x_train, y_train)

# # Save the preprocessor
# joblib.dump(preprocessor, 'preprocessor.pkl')

# # Save the model
# joblib.dump(model, 'heart_disease_model.pkl')
x_train.head()
