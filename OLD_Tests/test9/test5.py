import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

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

# Neural Network Model
nn_model = Sequential([
    Dense(512, input_shape=(X_train.shape[1],)),
    LayerNormalization(),
    LeakyReLU(),
    Dropout(0.4),
    Dense(256),
    LayerNormalization(),
    LeakyReLU(),
    Dropout(0.3),
    Dense(128),
    LayerNormalization(),
    LeakyReLU(),
    Dropout(0.2),
    Dense(64),
    LayerNormalization(),
    LeakyReLU(),
    Dense(1, activation='linear')
])

nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mean_squared_error'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = nn_model.fit(X_train, y_train, epochs=300, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=1)

# Decision Tree Model
tree_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
tree_model.fit(X_train, y_train)

# Make predictions
nn_preds = nn_model.predict(X_test).flatten()
tree_preds = tree_model.predict(X_test)

# Combine predictions using a meta-model (simple weighted average for now)
ensemble_preds = 0.5 * nn_preds + 0.5 * tree_preds

# Evaluate the ensemble model
ensemble_mse = mean_squared_error(y_test, ensemble_preds)
print(f"Test Mean Squared Error (Ensemble): {ensemble_mse}")

# Optional: Meta-model approach for better ensemble (Stacking)
from sklearn.linear_model import LinearRegression

# Combine predictions as features for the meta-model
meta_X_train = np.column_stack((nn_model.predict(X_train).flatten(), tree_model.predict(X_train)))
meta_X_val = np.column_stack((nn_model.predict(X_val).flatten(), tree_model.predict(X_val)))
meta_X_test = np.column_stack((nn_preds, tree_preds))

meta_model = LinearRegression()
meta_model.fit(meta_X_train, y_train)

meta_preds = meta_model.predict(meta_X_test)
meta_mse = mean_squared_error(y_test, meta_preds)
print(f"Test Mean Squared Error (Meta-Model Ensemble): {meta_mse}")

# Plotting results (Optional)
import matplotlib.pyplot as plt

plt.bar(['Neural Network', 'Gradient Boosting', 'Simple Ensemble', 'Meta-Model Ensemble'], [
    mean_squared_error(y_test, nn_preds),
    mean_squared_error(y_test, tree_preds),
    ensemble_mse,
    meta_mse
])
plt.ylabel('Test Mean Squared Error')
plt.title('Comparison of Different Approaches')
plt.show()
