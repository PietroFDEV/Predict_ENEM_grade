import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load the dataset
df = pd.read_csv('treinamento_alunos.csv')

# Define the features and target
features = ['Q001', 'Q002', 'Q003', 'Q004', 'Q005', 'Q006', 'Q007', 
            'Q008', 'Q009', 'Q010', 'Q011', 'Q012', 'Q013', 'Q014', 
            'Q015', 'Q016', 'Q017', 'Q018', 'Q019', 'Q020', 'Q021', 
            'Q022', 'Q023', 'Q024', 'Q025']
X = df[features]
y = df['NU_NOTA_REDACAO']

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), features)
    ])

# Define the Random Forest model pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train the model
model_pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = model_pipeline.predict(X_test)
test_mae = mean_absolute_error(y_test, y_pred)
print(f'Test MAE: {test_mae}')

# Load the new data
new_data = pd.read_csv('teste_alunos.csv')

# Preprocess the new data and make predictions
new_data_processed = preprocessor.transform(new_data[features])
predictions = model_pipeline.named_steps['regressor'].predict(new_data_processed)

# Add the predictions to the new data
new_data['NU_NOTA_REDACAO'] = predictions

# Save the updated data to a new CSV file
new_data.to_csv('teste_alunos_predicted.csv', index=False)
