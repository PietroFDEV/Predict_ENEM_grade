import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
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
numeric_features = data.select_dtypes(include=['number']).columns
categorical_features = data.select_dtypes(include=['object']).columns

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
    x = Dense(512, activation=None)(x)  # Ensure dimensions match
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.4)(x)
    x = Dense(512, activation=None)(x)  # Ensure dimensions match
    x = BatchNormalization()(x)
    x = Add()([x, residual])
    x = LeakyReLU()(x)
    
    # Residual Block 2
    residual = Dense(256, activation=None)(x)  # Adjust residual dimensions
    x = Dense(256, activation=None)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation=None)(x)
    x = BatchNormalization()(x)
    x = Add()([x, residual])
    x = LeakyReLU()(x)
    
    # Additional Block 3
    residual = Dense(128, activation=None)(x)  # Adjust residual dimensions
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

# Random Forest Model
rf_model = RandomForestRegressor(n_estimators=1000, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
resnet_preds = resnet_model.predict(X_test).flatten()
tree_preds = tree_model.predict(X_test)
rf_preds = rf_model.predict(X_test)

# Combine predictions using a meta-model (simple weighted average for now)
ensemble_preds = 0.5 * resnet_preds + 0.3 * tree_preds + 0.2 * rf_preds

# Evaluate the ensemble model
ensemble_mse = mean_squared_error(y_test, ensemble_preds)
print(f"Test Mean Squared Error (Ensemble): {ensemble_mse}")

# Optional: Meta-model approach for better ensemble (Stacking)
stacked_model = StackingRegressor(
    estimators=[
        ('resnet', GradientBoostingRegressor(n_estimators=1000, learning_rate=0.01, max_depth=6, random_state=42)),
        ('tree', RandomForestRegressor(n_estimators=1000, random_state=42))
    ],
    final_estimator=Ridge(alpha=1.0)
)

stacked_model.fit(X_train, y_train)
stacked_preds = stacked_model.predict(X_test)
stacked_mse = mean_squared_error(y_test, stacked_preds)
print(f"Test Mean Squared Error (Stacked Ensemble): {stacked_mse}")

# Plotting results (Optional)
plt.bar(['ResNet', 'Gradient Boosting', 'Random Forest', 'Simple Ensemble', 'Stacked Ensemble'], [
    mean_squared_error(y_test, resnet_preds),
    mean_squared_error(y_test, tree_preds),
    mean_squared_error(y_test, rf_preds),
    ensemble_mse,
    stacked_mse
])
plt.ylabel('Test Mean Squared Error')
plt.title('Comparison of Different Approaches')
plt.show()
