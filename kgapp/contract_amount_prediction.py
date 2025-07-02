import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.parse import unquote
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Function to decode URL-encoded strings
def decode_url(text):
    return unquote(text)

# Load the dataset
df = pd.read_csv('kgapp/datasets/datasets_converted.csv', encoding='utf-8')

# Decode Institution and Supplier columns
df['Institution'] = df['Institution'].apply(decode_url)
df['Supplier'] = df['Supplier'].apply(decode_url)

# Prepare features
le_institution = LabelEncoder()
le_supplier = LabelEncoder()
tfidf = TfidfVectorizer(max_features=1000)

# Function for safe label transformation
def safe_transform(encoder, value):
    if value not in encoder.classes_:
        encoder.classes_ = np.append(encoder.classes_, value)
    return encoder.transform([value])[0]

# Encode categorical variables and extract TF-IDF features
df['Institution_encoded'] = le_institution.fit_transform(df['Institution'])
df['Supplier_encoded'] = le_supplier.fit_transform(df['Supplier'])
contract_tfidf = tfidf.fit_transform(df['Contract'])

# Combine features
X = pd.concat([
    df[['Institution_encoded', 'Supplier_encoded']],
    pd.DataFrame(contract_tfidf.toarray())
], axis=1)

y = df['Amount']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.columns = X_train.columns.astype(str)
X_test.columns = X_test.columns.astype(str)

# Create and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Function to predict contract amount
def predict_contract_amount(institution, contract_description, supplier):
    institution_encoded = safe_transform(le_institution, institution)
    supplier_encoded = safe_transform(le_supplier, supplier)
    contract_transform = tfidf.transform([contract_description])

    input_data = pd.concat([
        pd.DataFrame([[institution_encoded, supplier_encoded]]),
        pd.DataFrame(contract_transform.toarray())
    ], axis=1)

    predicted_amount = model.predict(input_data)[0]
    return predicted_amount

# Function to get unique institutions and suppliers
def get_unique_values():
    institutions = df['Institution'].unique().tolist()
    suppliers = df['Supplier'].unique().tolist()
    return institutions, suppliers

# Function to visualize historical trends
def visualize_historical_trends(institution):
    institution_data = df[df['Institution'] == institution]
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=institution_data, x='Date', y='Amount', marker='o')
    plt.title(f'Historical trend for {institution}')
    plt.xlabel('Date')
    plt.ylabel('Sum')
    plt.xticks(rotation=45)
    plt.tight_layout()
    os.makedirs('../data/', exist_ok=True)
    plt.savefig('data/historical_trend.png')
    plt.close()

# Function to analyze factors affecting contract amount
def analyze_factors():
    feature_importance = model.feature_importances_
    features = X.columns

    feature_names = {
        '0': 'Institution',
        '1': 'Supplier',
    }
    for i in range(2, len(features)):
        feature_names[str(i)] = f'Word_{i-1}'

    importance_df = pd.DataFrame({
        'feature': [feature_names.get(str(f), f) for f in features],
        'importance': feature_importance
    })
    importance_df = importance_df.sort_values('importance', ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importance_df)
    plt.title('Top 10 factors influencing contract amount')
    plt.xlabel('Importance')
    plt.ylabel('Factor')
    plt.tight_layout()
    os.makedirs('../data/', exist_ok=True)
    plt.savefig('data/factor_importance.png')
    plt.close()

# Function to evaluate the model's performance
def evaluate_model(X_train, X_test, y_train, y_test, model):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    mse_train = mean_squared_error(y_train, y_train_pred)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    r2_train = r2_score(y_train, y_train_pred)

    mse_test = mean_squared_error(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)

    print(f'Training Metrics: MSE = {mse_train:.2f}, MAE = {mae_train:.2f}, R² = {r2_train:.2f}')
    print(f'Testing Metrics: MSE = {mse_test:.2f}, MAE = {mae_test:.2f}, R² = {r2_test:.2f}')

    metrics = {
        'Metric': ['MSE', 'MAE', 'R²'],
        'Train': [mse_train, mae_train, r2_train],
        'Test': [mse_test, mae_test, r2_test]
    }
    metrics_df = pd.DataFrame(metrics)

    os.makedirs('../results/', exist_ok=True)
    metrics_df.to_csv('../results/model_performance.csv', index=False)

    return metrics_df

# Evaluate the model
evaluate_model(X_train, X_test, y_train, y_test, model)


# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.feature_extraction.text import TfidfVectorizer
# import matplotlib.pyplot as plt
# import seaborn as sns
# from urllib.parse import unquote
# import os
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#
#
# # Function to decode URL-encoded strings
# def decode_url(text):
#     return unquote(text)
#
#
# # Load the dataset
# df = pd.read_csv('kgapp/datasets/datasets_converted.csv', encoding='utf-8')
#
# # Decode Institution and Supplier columns
# df['Institution'] = df['Institution'].apply(decode_url)
# df['Supplier'] = df['Supplier'].apply(decode_url)
#
# # Prepare features
# le_institution = LabelEncoder()
# le_supplier = LabelEncoder()
# tfidf = TfidfVectorizer(max_features=2000)  # Increase the number of features for TF-IDF
#
#
# # Function for safe label transformation
# def safe_transform(encoder, value):
#     if value not in encoder.classes_:
#         encoder.classes_ = np.append(encoder.classes_, value)
#     return encoder.transform([value])[0]
#
#
# # Encode categorical variables and extract TF-IDF features
# df['Institution_encoded'] = le_institution.fit_transform(df['Institution'])
# df['Supplier_encoded'] = le_supplier.fit_transform(df['Supplier'])
# contract_tfidf = tfidf.fit_transform(df['Contract'])
#
# # Combine features
# X = pd.concat([
#     df[['Institution_encoded', 'Supplier_encoded']],
#     pd.DataFrame(contract_tfidf.toarray())
# ], axis=1)
#
# y = df['Amount']
#
# # Handle missing values if present (fill with mean for simplicity)
# X = X.fillna(X.mean())
# y = y.fillna(y.mean())
#
# X.columns = X.columns.astype(str)
#
# # Convert data to float for scaling
# X = X.astype(float)
#
# # Initialize the scaler
# scaler = StandardScaler()
#
# # Fit and transform the features
# X_scaled = scaler.fit_transform(X)
#
# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
#
# # Create and train the model with hyperparameter tuning
# model = RandomForestRegressor(random_state=42)
#
# # Hyperparameter tuning with GridSearchCV
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }
#
# grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
# grid_search.fit(X_train, y_train)
#
# # Best model
# model = grid_search.best_estimator_
#
# # Evaluate the model
# def evaluate_model(X_train, X_test, y_train, y_test, model):
#     # Predict on training and test data
#     y_train_pred = model.predict(X_train)
#     y_test_pred = model.predict(X_test)
#
#     # Calculate the metrics
#     mse_train = mean_squared_error(y_train, y_train_pred)
#     mae_train = mean_absolute_error(y_train, y_train_pred)
#     r2_train = r2_score(y_train, y_train_pred)
#
#     mse_test = mean_squared_error(y_test, y_test_pred)
#     mae_test = mean_absolute_error(y_test, y_test_pred)
#     r2_test = r2_score(y_test, y_test_pred)
#
#     print(f'Training Metrics: MSE = {mse_train:.2f}, MAE = {mae_train:.2f}, R² = {r2_train:.2f}')
#     print(f'Testing Metrics: MSE = {mse_test:.2f}, MAE = {mae_test:.2f}, R² = {r2_test:.2f}')
#
#     # Create a DataFrame with the results
#     metrics = {
#         'Metric': ['MSE', 'MAE', 'R²'],
#         'Train': [mse_train, mae_train, r2_train],
#         'Test': [mse_test, mae_test, r2_test]
#     }
#     metrics_df = pd.DataFrame(metrics)
#
#     # Save the results to a CSV file
#     os.makedirs('../results/', exist_ok=True)
#     metrics_df.to_csv('../results/model_performance.csv', index=False)
#
#     return metrics_df
#
#
# # Evaluate the improved model
# evaluate_model(X_train, X_test, y_train, y_test, model)
