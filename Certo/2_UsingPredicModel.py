### Alunos:
### Pietro Goudel Favoreto
### Lucas Luan Morais Martins

import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load the preprocessing objects
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')
numeric_features = joblib.load('numeric_features.pkl')
categorical_features = joblib.load('categorical_features.pkl')

# Load the trained models
resnet_model = load_model('resnet_model.h5')
rf_model = joblib.load('rf_model.pkl')
gb_model = joblib.load('gb_model.pkl')
stacked_model = joblib.load('stacked_model.pkl')

# Custom LabelEncoder to handle unseen labels
class CustomLabelEncoder:
    def __init__(self):
        self.classes_ = {}
        self.classes_reverse_ = {}
    
    def fit(self, y):
        self.classes_ = {cls: idx for idx, cls in enumerate(np.unique(y))}
        self.classes_reverse_ = {idx: cls for cls, idx in self.classes_.items()}
        return self
    
    def transform(self, y):
        return np.array([self.classes_.get(cls, -1) for cls in y])
    
    def fit_transform(self, y):
        return self.fit(y).transform(y)
    
    def inverse_transform(self, y):
        return np.array([self.classes_reverse_.get(idx, None) for idx in y])

# Update label encoders with custom label encoder
for feature in categorical_features:
    le = CustomLabelEncoder()
    le.classes_ = {k: v for k, v in enumerate(label_encoders[feature].classes_)}
    label_encoders[feature] = le

# Preprocessing function for new data
def preprocess_new_data(new_data):
    new_data[numeric_features] = new_data[numeric_features].fillna(new_data[numeric_features].mean())
    new_data[categorical_features] = new_data[categorical_features].fillna(new_data[categorical_features].mode().iloc[0])
    
    for feature in categorical_features:
        le = label_encoders[feature]
        new_data[feature] = le.transform(new_data[feature].astype(str))
    
    new_X_scaled = scaler.transform(new_data)
    return new_X_scaled

# Load new data
new_data = pd.read_csv('teste_alunos.csv')

# Preprocess the new data
new_X_scaled = preprocess_new_data(new_data)

# Make predictions using the stacked ensemble model
new_stacked_preds = stacked_model.predict(new_X_scaled)

# Round predictions and convert to integers
new_stacked_preds = np.round(new_stacked_preds).astype(int)

# Add the predictions to the new data
new_data['NU_NOTA_REDACAO'] = new_stacked_preds

# Save the updated data to a new CSV file
new_data.to_csv('teste_alunos_with_predictions.csv', index=False)


### Alunos:
### Pietro Goudel Favoreto
### Lucas Luan Morais Martins