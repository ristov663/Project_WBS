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
    sns.scatterplot(data=institution_data, x='Date', y='Amount')
    plt.title(f'Историски тренд на договори за {institution}')
    plt.xlabel('Датум')
    plt.ylabel('Сума')
    plt.xticks(rotation=45)
    plt.tight_layout()
    os.makedirs('../data/', exist_ok=True)
    plt.savefig('data/historical_trend.png')
    plt.close()


# Function to analyze factors affecting contract amount
def analyze_factors():
    feature_importance = model.feature_importances_
    features = X.columns

    # Create a dictionary to map indices to descriptive names
    feature_names = {
        '0': 'Institution',
        '1': 'Supplier',
    }
    for i in range(2, len(features)):
        feature_names[str(i)] = f'Word_{i-1}'

    # Create DataFrame with named features
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
