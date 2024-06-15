import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the dataset
logging.info("Loading training dataset...")
df = pd.read_csv('treinamento_alunos.csv')

# Define the features and target
features = ['Q001', 'Q002', 'Q003', 'Q004', 'Q005', 'Q006', 'Q007', 
            'Q008', 'Q009', 'Q010', 'Q011', 'Q012', 'Q013', 'Q014', 
            'Q015', 'Q016', 'Q017', 'Q018', 'Q019', 'Q020', 'Q021', 
            'Q022', 'Q023', 'Q024', 'Q025']
X = df[features]
y = df['NU_NOTA_REDACAO']

# Split the data into training, validation, and test sets
logging.info("Splitting the dataset into train, validation, and test sets...")
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Preprocessing pipeline
logging.info("Setting up preprocessing pipeline...")
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), features)
    ])

# Fit and transform the data
X_train = preprocessor.fit_transform(X_train)
X_val = preprocessor.transform(X_val)
X_test = preprocessor.transform(X_test)

# Define the model
logging.info("Defining the neural network model...")
model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(1)
])

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

# Callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

# Train the model
logging.info("Training the model...")
history = model.fit(X_train, y_train, epochs=200, batch_size=64, validation_data=(X_val, y_val),
                    callbacks=[early_stopping, reduce_lr])

# Evaluate the model
logging.info("Evaluating the model...")
test_loss, test_mae = model.evaluate(X_test, y_test)
logging.info(f'Test MAE: {test_mae}')

# Load the new data
logging.info("Loading new data for predictions...")
new_data = pd.read_csv('teste_alunos.csv')

# Preprocess the new data
logging.info("Preprocessing new data and making predictions...")
new_data_processed = preprocessor.transform(new_data[features])

# Make predictions
predictions = model.predict(new_data_processed)

# Add the predictions to the new data
logging.info("Adding predictions to the new data...")
new_data['NU_NOTA_REDACAO'] = predictions

# Save the updated data to a new CSV file
logging.info("Saving the predictions to a new CSV file...")
new_data.to_csv('teste_alunos_predicted.csv', index=False)
logging.info("Done!")
