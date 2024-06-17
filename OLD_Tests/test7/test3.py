import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder

# Load the data
data = pd.read_csv('treinamento_alunos.csv')

# Remove rows with missing target values
data = data.dropna(subset=['NU_NOTA_REDACAO'])

# Separate features and target
X = data.drop(columns=['NU_NOTA_REDACAO'])
y = data['NU_NOTA_REDACAO']

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Handle categorical columns (for this example, dropping them)
X = X.drop(columns=categorical_cols)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model definition
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf.fit(X_train, y_train)

# Predict on test set
y_pred = rf.predict(X_test)

# Calculate Mean Squared Error
test_mse = mean_squared_error(y_test, y_pred)
print(f"Test Mean Squared Error (Approach 3 - Random Forest): {test_mse}")
