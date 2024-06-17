import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.metrics import mean_absolute_error
import logging
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Configurando o logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Função para criar o modelo de rede neural
def create_nn_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

# Carregando o dataset
logging.info("Carregando o dataset de treinamento...")
df = pd.read_csv('treinamento_alunos.csv')

# Definindo as features e o target
features = ['Q001', 'Q002', 'Q003', 'Q004', 'Q005', 'Q006', 'Q007', 
            'Q008', 'Q009', 'Q010', 'Q011', 'Q012', 'Q013', 'Q014', 
            'Q015', 'Q016', 'Q017', 'Q018', 'Q019', 'Q020', 'Q021', 
            'Q022', 'Q023', 'Q024', 'Q025']
X = df[features]
y = df['NU_NOTA_REDACAO']

# Dividindo os dados em conjuntos de treinamento, validação e teste
logging.info("Dividindo o dataset em conjuntos de treinamento, validação e teste...")
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Pipeline de pré-processamento
logging.info("Configurando o pipeline de pré-processamento...")
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), features)
    ])

# Ajustando o pré-processador nos dados de treinamento
logging.info("Ajustando o pré-processador...")
preprocessor.fit(X_train)

# Transformando os dados
X_train_transformed = preprocessor.transform(X_train)
X_val_transformed = preprocessor.transform(X_val)
X_test_transformed = preprocessor.transform(X_test)

# Definindo e ajustando o modelo Decision Tree com RandomizedSearchCV
logging.info("Ajustando o Decision Tree Regressor...")
dt_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', DecisionTreeRegressor(random_state=42))
])
dt_param_dist = {
    'regressor__max_depth': sp_randint(3, 10),
    'regressor__min_samples_split': sp_randint(2, 10),
    'regressor__min_samples_leaf': sp_randint(1, 10)
}
dt_random_search = RandomizedSearchCV(dt_model, param_distributions=dt_param_dist, n_iter=5, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=2)
dt_random_search.fit(X_train, y_train)
logging.info(f"Melhores parâmetros para Decision Tree: {dt_random_search.best_params_}")
dt_best_model = dt_random_search.best_estimator_
dt_val_pred = dt_best_model.predict(X_val)
dt_mae = mean_absolute_error(y_val, dt_val_pred)
logging.info(f'Decision Tree MAE: {dt_mae}')

# Definindo e ajustando o modelo Random Forest com RandomizedSearchCV
logging.info("Ajustando o Random Forest Regressor...")
rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])
rf_param_dist = {
    'regressor__n_estimators': sp_randint(50, 100),
    'regressor__max_depth': sp_randint(3, 10),
    'regressor__min_samples_split': sp_randint(2, 10),
    'regressor__min_samples_leaf': sp_randint(1, 10)
}
rf_random_search = RandomizedSearchCV(rf_model, param_distributions=rf_param_dist, n_iter=5, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=2)
rf_random_search.fit(X_train, y_train)
logging.info(f"Melhores parâmetros para Random Forest: {rf_random_search.best_params_}")
rf_best_model = rf_random_search.best_estimator_
rf_val_pred = rf_best_model.predict(X_val)
rf_mae = mean_absolute_error(y_val, rf_val_pred)
logging.info(f'Random Forest MAE: {rf_mae}')

# Definindo e ajustando o modelo Gradient Boosting com RandomizedSearchCV
logging.info("Ajustando o Gradient Boosting Regressor...")
gb_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(random_state=42))
])
gb_param_dist = {
    'regressor__n_estimators': sp_randint(50, 100),
    'regressor__learning_rate': sp_uniform(0.01, 0.1),
    'regressor__max_depth': sp_randint(3, 10),
    'regressor__min_samples_split': sp_randint(2, 10),
    'regressor__min_samples_leaf': sp_randint(1, 10)
}
gb_random_search = RandomizedSearchCV(gb_model, param_distributions=gb_param_dist, n_iter=5, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=2)
gb_random_search.fit(X_train, y_train)
logging.info(f"Melhores parâmetros para Gradient Boosting: {gb_random_search.best_params_}")
gb_best_model = gb_random_search.best_estimator_
gb_val_pred = gb_best_model.predict(X_val)
gb_mae = mean_absolute_error(y_val, gb_val_pred)
logging.info(f'Gradient Boosting MAE: {gb_mae}')

# Ajustando o modelo de Rede Neural com RandomizedSearchCV
logging.info("Ajustando o modelo de Rede Neural...")
input_dim = X_train_transformed.shape[1]
nn_model = KerasRegressor(model=create_nn_model, input_dim=input_dim, epochs=50, batch_size=32, verbose=0)
nn_param_dist = {
    'epochs': sp_randint(50, 100),
    'batch_size': sp_randint(16, 64),
    'model__optimizer': ['adam', 'rmsprop']
}
nn_random_search = RandomizedSearchCV(nn_model, param_distributions=nn_param_dist, n_iter=5, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=2)
nn_random_search.fit(X_train_transformed, y_train)
logging.info(f"Melhores parâmetros para Rede Neural: {nn_random_search.best_params_}")
nn_best_model = nn_random_search.best_estimator_
nn_val_pred = nn_best_model.predict(X_val_transformed)
nn_mae = mean_absolute_error(y_val, nn_val_pred)
logging.info(f'Neural Network MAE: {nn_mae}')

# Combinando modelos usando Stacking Regressor
logging.info("Ajustando o Stacking Regressor...")
estimators = [
    ('dt', dt_best_model),
    ('rf', rf_best_model),
    ('gb', gb_best_model),
    ('nn', nn_best_model)
]
stacking_model = StackingRegressor(
    estimators=estimators,
    final_estimator=GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
)
stacking_model.fit(X_train_transformed, y_train)
stacking_val_pred = stacking_model.predict(X_val_transformed)
stacking_mae = mean_absolute_error(y_val, stacking_val_pred)
logging.info(f'Stacking Regressor MAE: {stacking_mae}')

# Avaliando o melhor modelo no conjunto de teste
best_model = stacking_model
test_pred = best_model.predict(X_test_transformed)
test_mae = mean_absolute_error(y_test, test_pred)
logging.info(f'Test MAE: {test_mae}')

# Carregando os novos dados
logging.info("Carregando novos dados para previsões...")
new_data = pd.read_csv('teste_alunos.csv')

# Pré-processando os novos dados usando o pré-processador ajustado
logging.info("Pré-processando novos dados e fazendo previsões...")
new_data_processed = preprocessor.transform(new_data[features])

# Fazendo previsões nos novos dados
new_data['NU_NOTA_REDACAO'] = best_model.predict(new_data_processed)

# Salvando os novos dados com as previsões
new_data.to_csv('teste_alunos_predicted.csv', index=False)
logging.info("Previsões salvas em 'teste_alunos_predicted.csv'")
