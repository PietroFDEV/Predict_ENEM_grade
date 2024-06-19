import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU, Add
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('treinamento_alunos.csv')

# Separate numeric and categorical columns
numeric_features = data.select_dtypes(include=['number']).columns.tolist()
categorical_features = data.select_dtypes(include=['object']).columns.tolist()

# Remove target column from numeric features
numeric_features.remove('NU_NOTA_REDACAO')

# Fill missing values
data[numeric_features] = data[numeric_features].fillna(data[numeric_features].mean())
data[categorical_features] = data[categorical_features].fillna(data[categorical_features].mode().iloc[0])

# Encode categorical variables
label_encoders = {}
for feature in categorical_features:
    le = LabelEncoder()
    data[feature] = le.fit_transform(data[feature])
    label_encoders[feature] = le

# Separate features and target
X = data.drop(columns=['NU_NOTA_REDACAO'])
y = data['NU_NOTA_REDACAO']

# Normalize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Advanced Neural Network Model with Residual Connections
def build_resnet(input_shape):
    inputs = Input(shape=input_shape)
    x = Dense(512, activation=None)(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.5)(x)
    
    # Residual Block 1
    residual = x
    x = Dense(512, activation=None)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.4)(x)
    x = Dense(512, activation=None)(x)
    x = BatchNormalization()(x)
    x = Add()([x, residual])
    x = LeakyReLU()(x)
    
    # Residual Block 2
    residual = Dense(256, activation=None)(x)
    x = Dense(256, activation=None)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation=None)(x)
    x = BatchNormalization()(x)
    x = Add()([x, residual])
    x = LeakyReLU()(x)
    
    # Additional Block 3
    residual = Dense(128, activation=None)(x)
    x = Dense(128, activation=None)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation=None)(x)
    x = BatchNormalization()(x)
    x = Add()([x, residual])
    x = LeakyReLU()(x)
    
    outputs = Dense(1, activation='linear')(x)
    
    model = Model(inputs, outputs)
    return model

# Create and compile the ResNet model
resnet_model = build_resnet((X_train.shape[1],))
optimizer = Adam(learning_rate=0.001)
resnet_model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_squared_error'])

early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)

history = resnet_model.fit(X_train, y_train, epochs=500, batch_size=32, validation_data=(X_val, y_val),
                           callbacks=[early_stopping, reduce_lr], verbose=1)

# Decision Tree Model
tree_model = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.01, max_depth=6, random_state=42)
tree_model.fit(X_train, y_train)

# Meta-model approach for better ensemble (Stacking)
stacked_model = StackingRegressor(
    estimators=[
        ('resnet', GradientBoostingRegressor(n_estimators=1000, learning_rate=0.01, max_depth=6, random_state=42)),
        ('tree', GradientBoostingRegressor(n_estimators=1000, learning_rate=0.01, max_depth=6, random_state=42))
    ],
    final_estimator=Ridge(alpha=1.0)
)

stacked_model.fit(X_train, y_train)
stacked_preds = stacked_model.predict(X_test)
stacked_mse = mean_squared_error(y_test, stacked_preds)
print(f"Test Mean Squared Error (Stacked Ensemble): {stacked_mse}")

# Preprocessing function for new data
def preprocess_new_data(new_data):
    new_data[numeric_features] = new_data[numeric_features].fillna(new_data[numeric_features].mean())
    new_data[categorical_features] = new_data[categorical_features].fillna(new_data[categorical_features].mode().iloc[0])
    
    for feature in categorical_features:
        le = label_encoders[feature]
        new_data[feature] = le.transform(new_data[feature])
    
    new_X_scaled = scaler.transform(new_data)
    return new_X_scaled

# Load new data
new_data = pd.read_csv('teste_alunos.csv')

# Preprocess the new data
new_X_scaled = preprocess_new_data(new_data)

# Make predictions using the stacked ensemble model
new_stacked_preds = stacked_model.predict(new_X_scaled)

# Add the predictions to the new data
new_data['NU_NOTA_REDACAO'] = new_stacked_preds

# Save the updated data to a new CSV file
new_data.to_csv('teste_alunos_with_predictions.csv', index=False)

# Plotting results (Optional)
plt.bar(['ResNet', 'Gradient Boosting', 'Stacked Ensemble'], [
    mean_squared_error(y_test, resnet_model.predict(X_test).flatten()),
    mean_squared_error(y_test, tree_model.predict(X_test)),
    stacked_mse
])
plt.ylabel('Test Mean Squared Error')
plt.title('Comparison of Different Approaches')
plt.show()
