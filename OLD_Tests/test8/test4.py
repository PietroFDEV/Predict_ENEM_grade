import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout
import matplotlib.pyplot as plt
import time  # Import time module for measuring execution time

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

# Approach 1: Adjusting Model Architecture and Hyperparameters
input_shape = X_train.shape[1]

inputs = Input(shape=(input_shape,))
x = Dense(512, activation='relu')(inputs)
x = Dropout(0.4)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(64, activation='relu')(x)
outputs = Dense(1, activation='linear')(x)

model1 = Model(inputs=inputs, outputs=outputs)

model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mean_squared_error'])

print("Training Model 1...")
start_time = time.time()  # Measure start time
history1 = model1.fit(X_train, y_train, epochs=300, batch_size=32, validation_data=(X_val, y_val), verbose=0)
end_time = time.time()  # Measure end time
print(f"Model 1 trained in {end_time - start_time} seconds")

print("Evaluating Model 1...")
test_loss1, test_mse1 = model1.evaluate(X_test, y_test)
print(f"Test Mean Squared Error (Approach 1): {test_mse1}")

# Approach 2: Feature Engineering and Data Preprocessing
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_scaled)

X_train_poly, X_temp_poly, y_train_poly, y_temp_poly = train_test_split(X_poly, y, test_size=0.3, random_state=42)
X_val_poly, X_test_poly, y_val_poly, y_test_poly = train_test_split(X_temp_poly, y_temp_poly, test_size=0.5, random_state=42)

inputs2 = Input(shape=(X_train_poly.shape[1],))
x = Dense(512, activation='relu')(inputs2)
x = Dropout(0.4)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(64, activation='relu')(x)
outputs2 = Dense(1, activation='linear')(x)

model2 = Model(inputs=inputs2, outputs=outputs2)

model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mean_squared_error'])

print("Training Model 2...")
start_time = time.time()  # Measure start time
history2 = model2.fit(X_train_poly, y_train_poly, epochs=300, batch_size=32, validation_data=(X_val_poly, y_val_poly), verbose=0)
end_time = time.time()  # Measure end time
print(f"Model 2 trained in {end_time - start_time} seconds")

print("Evaluating Model 2...")
test_loss2, test_mse2 = model2.evaluate(X_test_poly, y_test_poly)
print(f"Test Mean Squared Error (Approach 2): {test_mse2}")

# Approach 3: Ensemble Methods (Random Forest)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
print("Training Random Forest...")
start_time = time.time()  # Measure start time
rf.fit(X_train, y_train)
end_time = time.time()  # Measure end time
print(f"Random Forest trained in {end_time - start_time} seconds")

y_pred_rf = rf.predict(X_test)
test_mse3 = mean_squared_error(y_test, y_pred_rf)
print(f"Test Mean Squared Error (Approach 3 - Random Forest): {test_mse3}")

# Plotting results (Optional)
plt.bar(['Approach 1', 'Approach 2', 'Approach 3 (RF)'], [test_mse1, test_mse2, test_mse3])
plt.ylabel('Test Mean Squared Error')
plt.title('Comparison of Different Approaches')
plt.show()
