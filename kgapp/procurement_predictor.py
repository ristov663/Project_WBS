# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from sentence_transformers import SentenceTransformer
# from sklearn.neighbors import NearestNeighbors
# import re
#
# # 1. Параметри
# TIME_WINDOW = 365
# TOP_K = 10
# EMBEDDING_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'
#
#
# # 2. Подготовка на податоци
# def parse_dates(date_str):
#     try:
#         return datetime.strptime(date_str, '%Y-%m-%d')
#     except:
#         return datetime.now() - timedelta(days=365 * 5)
#
#
# def clean_text(text):
#     text = re.sub(r'[^\w\s]', '', str(text))
#     return text.lower().strip()
#
#
# # 3. Вчитување и филтрирање
# df = pd.read_csv('kgapp/datasets/datasets_converted.csv', parse_dates=['Date'])
# df['Contract'] = df['Contract'].apply(clean_text)
# df = df[(df['Amount'] > 0) & (df['Contract'].str.len() > 10)].dropna(subset=['Contract', 'Date', 'Amount'])
#
# # 4. Проверка на податоците
# if len(df) == 0:
#     raise ValueError("Нема достапни податоци после филтрирање!")
#
# # 5. Адаптивна поделба
# split_date = df['Date'].quantile(0.8, interpolation='nearest')  # 80% за тренирање
# train = df[df['Date'] < split_date]
# test = df[df['Date'] >= split_date]
#
# if len(test) == 0:
#     raise ValueError("Тест сетот е празен. Променете го датумот на поделба.")
#
# # 6. Генерирање embeddings
# model = SentenceTransformer(EMBEDDING_MODEL)
# train_embeddings = model.encode(train['Contract'].tolist())
# test_embeddings = model.encode(test['Contract'].tolist())
#
# # 7. Nearest Neighbors модел
# nn = NearestNeighbors(n_neighbors=TOP_K + 1, metric='cosine')
# nn.fit(train_embeddings)
#
#
# # 8. Временски филтер
# def filter_by_date(query_date, candidates):
#     valid = []
#     for idx in candidates:
#         delta = abs((query_date - train.iloc[idx]['Date']).days)
#         if delta <= TIME_WINDOW:
#             valid.append(idx)
#     return valid[:TOP_K] if valid else []
#
#
# # 9. Предвидување
# train_preds, train_truths = [], []
# preds, truths = [], []
#
# # За Training сетот
# for emb, date, true_amount in zip(train_embeddings, train['Date'], train['Amount']):
#     _, indices = nn.kneighbors([emb])
#     valid_indices = filter_by_date(date, indices[0][1:])
#     pred_amount = train.iloc[valid_indices]['Amount'].median() if valid_indices else train['Amount'].median()
#     train_preds.append(pred_amount)
#     train_truths.append(true_amount)
#
# # За Testing сетот
# for emb, date, true_amount in zip(test_embeddings, test['Date'], test['Amount']):
#     _, indices = nn.kneighbors([emb])
#     valid_indices = filter_by_date(date, indices[0][1:])
#     pred_amount = train.iloc[valid_indices]['Amount'].median() if valid_indices else train['Amount'].median()
#     preds.append(pred_amount)
#     truths.append(true_amount)
#
# # 10. Евалуација
# if len(train_preds) > 0 and len(preds) > 0:
#     print("--- Training Set Metrics ---")
#     print(f"MAE: {mean_absolute_error(train_truths, train_preds):.2f}")
#     print(f"MSE: {mean_squared_error(train_truths, train_preds):.2f}")
#     print(f"R²: {r2_score(train_truths, train_preds):.2f}")
#
#     print("\n--- Testing Set Metrics ---")
#     print(f"MAE: {mean_absolute_error(truths, preds):.2f}")
#     print(f"MSE: {mean_squared_error(truths, preds):.2f}")
#     print(f"R²: {r2_score(truths, preds):.2f}")
#     print(f"Median Baseline: {np.median(train['Amount']):.2f}")
# else:
#     print("Нема предвидувања за евалуација.")
#
#
# # 11. Пример за употреба
# def predict_amount(contract_text, date=datetime.now()):
#     emb = model.encode([clean_text(contract_text)])
#     _, indices = nn.kneighbors(emb)
#     valid_indices = filter_by_date(date, indices[0][1:])
#     return train.iloc[valid_indices]['Amount'].median() if valid_indices else train['Amount'].median()
#
#
# # Тестирање
# print("\nТест предвидување:", predict_amount("Набавка на здравствена опрема", datetime(2024, 3, 1)))





import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine
from sklearn.preprocessing import RobustScaler
import re
import joblib

# 1. Параметри
TIME_WINDOW = 180  # 6 месеци
TOP_K = 15  # Број на соседи
EMBEDDING_MODEL = 'paraphrase-multilingual-mpnet-base-v2'  # Подобар модел
PCA_COMPONENTS = 128  # Намалување на димензии


# 2. Подготовка на податоци
def parse_dates(date_str):
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except:
        return pd.NaT


def clean_text(text):
    text = re.sub(r'\d+', '', str(text))  # Отстрани броеви
    text = re.sub(r'\b\w{1,3}\b', '', text)  # Отстрани кратки зборови
    return text.lower().strip()


# 3. Вчитување и филтрирање
df = pd.read_csv('kgapp/datasets/datasets_converted.csv', parse_dates=['Date'])
df = df[(df['Amount'] > 1e4) & (df['Amount'] < 1e9)]  # Филтер за outliers
df['Contract'] = df['Contract'].apply(clean_text)
df = df[df['Contract'].str.len() > 50].dropna(subset=['Contract', 'Date', 'Amount'])

# 4. Временска анализа
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

df['Days'] = (datetime.now() - df['Date']).dt.days

# 5. Поделба со стратификација
train = df[df['Year'] < 2023]
test = df[df['Year'] >= 2023]

if len(test) == 0:
    latest_year = df['Year'].max() - 1
    train = df[df['Year'] <= latest_year]
    test = df[df['Year'] > latest_year]

# 6. Генерирање embeddings со PCA
model = SentenceTransformer(EMBEDDING_MODEL)
train_embeddings = model.encode(train['Contract'].tolist())
test_embeddings = model.encode(test['Contract'].tolist())

pca = PCA(n_components=PCA_COMPONENTS)
train_embeddings = pca.fit_transform(train_embeddings)
test_embeddings = pca.transform(test_embeddings)

# 7. Скалирање на карактеристики
scaler = RobustScaler()
train_scaled = scaler.fit_transform(np.hstack([train_embeddings, train[['Year', 'Month']]]))
test_scaled = scaler.transform(np.hstack([test_embeddings, test[['Year', 'Month']]]))

# 8. Модел за пребарување
nn = NearestNeighbors(n_neighbors=TOP_K * 2, metric='cosine', n_jobs=-1)
nn.fit(train_scaled)

# 9. Напредно предвидување
def weighted_prediction(indices, distances, query_date):
    weights = 1 / (np.array(distances) + 1e-6)
    valid = []

    for i, idx in enumerate(indices):
        delta_days = abs((query_date - train.iloc[idx]['Date']).days)
        if delta_days > TIME_WINDOW:
            continue
        time_weight = np.exp(-delta_days / TIME_WINDOW)
        valid.append((train.iloc[idx]['Amount'], weights[i] * time_weight))

    if not valid:
        return None

    values, weights = zip(*valid)
    return np.average(values[:TOP_K], weights=weights[:TOP_K])

# 10. Евалуација на Training сетот
train_preds, train_truths = [], []
for i, (emb, date) in enumerate(zip(train_scaled, train['Date'])):
    distances, indices = nn.kneighbors([emb])
    pred = weighted_prediction(indices[0][1:], distances[0][1:], date)
    if pred is None:
        pred = train['Amount'].median()
    train_preds.append(pred)
    train_truths.append(train.iloc[i]['Amount'])

# 11. Евалуација на Testing сетот
test_preds, test_truths = [], []
for i, (emb, date) in enumerate(zip(test_scaled, test['Date'])):
    distances, indices = nn.kneighbors([emb])
    pred = weighted_prediction(indices[0][1:], distances[0][1:], date)
    if pred is None:
        pred = train['Amount'].median()
    test_preds.append(pred)
    test_truths.append(test.iloc[i]['Amount'])

# 12. Метрики за Training
print("Training Set Metrics:")
print(f"MAE: {mean_absolute_error(train_truths, train_preds):,.2f}")
print(f"MSE: {mean_squared_error(train_truths, train_preds):,.2f}")
print(f"R²: {r2_score(train_truths, train_preds):.2f}\n")

# 13. Метрики за Testing
print("Testing Set Metrics:")
print(f"MAE: {mean_absolute_error(test_truths, test_preds):,.2f}")
print(f"MSE: {mean_squared_error(test_truths, test_preds):,.2f}")
print(f"R²: {r2_score(test_truths, test_preds):.2f}")

# 14. Зачувување на моделот
joblib.dump({'nn': nn, 'pca': pca, 'scaler': scaler}, 'procurement_model.pkl')

# 15. Пример за употреба
def predict_amount(contract_text, date=datetime.now()):
    emb = model.encode([clean_text(contract_text)])
    emb_pca = pca.transform(emb)
    emb_scaled = scaler.transform(np.hstack([emb_pca, [[date.year, date.month]]]))
    distances, indices = nn.kneighbors(emb_scaled)
    return weighted_prediction(indices[0][1:], distances[0][1:], date) or train['Amount'].median()

print("\nТест предвидување:", round(predict_amount("Набавка на дигитална здравствена опрема", datetime(2024, 3, 1)), 2))






# import pandas as pd
# import numpy as np
# from datetime import datetime
# import re
# import torch
# import joblib
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import FunctionTransformer, StandardScaler, RobustScaler
# from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
# from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
# from transformers import AutoTokenizer, AutoModel
# from xgboost import XGBRegressor
# from category_encoders import TargetEncoder
#
# # Конфигурација
# CONFIG = {
#     "text_model": "EMBEDDIA/sloberta",
#     "max_text_length": 512,
#     "time_window": 180,
#     "quantile_range": (0.05, 0.95),
#     "test_size": 0.2,
#     "device": "cuda" if torch.cuda.is_available() else "cpu"
# }
#
#
# class DataPreprocessor:
#     def __init__(self):
#         self.tokenizer = AutoTokenizer.from_pretrained(CONFIG["text_model"])
#         self.text_model = AutoModel.from_pretrained(CONFIG["text_model"]).to(CONFIG["device"])
#
#     def clean_text(self, text):
#         text = re.sub(r'\d+|[^\w\s]', '', str(text).lower())
#         cleaned = ' '.join([word for word in text.split() if len(word) > 3])
#         return cleaned if len(cleaned) > 10 else "нема опис"
#
#     def generate_embeddings(self, texts):
#         texts = [str(t) for t in texts]  # Експлицитна конверзија во стринг
#         inputs = self.tokenizer(
#             texts,
#             return_tensors="pt",
#             padding=True,
#             truncation=True,
#             max_length=CONFIG["max_text_length"]
#         ).to(CONFIG["device"])
#
#         with torch.no_grad():
#             outputs = self.text_model(**inputs)
#
#         return outputs.last_hidden_state.mean(dim=1).cpu().numpy()
#
#     def add_temporal_features(self, df):
#         df['Year'] = df['Date'].dt.year
#         df['Quarter'] = df['Date'].dt.quarter
#         df['Days'] = (datetime.now() - df['Date']).dt.days
#         return df
#
#
# class ProcurementPredictor:
#     def __init__(self):
#         self.preprocessor = ColumnTransformer([
#             ('text', Pipeline([
#                 ('embed', FunctionTransformer(self._text_embeddings, validate=False)),
#                 ('scaler', StandardScaler())
#             ]), ['Contract']),
#
#             ('institution', TargetEncoder(), ['Institution']),
#             ('supplier', TargetEncoder(), ['Supplier']),
#             ('temporal', RobustScaler(), ['Year', 'Quarter', 'Days'])
#         ])
#
#         self.model = XGBRegressor(
#             objective='reg:absoluteerror',
#             tree_method='hist',
#             n_estimators=1000,
#             learning_rate=0.05,
#             random_state=42
#         )
#
#         self.pipeline = Pipeline([
#             ('preprocessor', self.preprocessor),
#             ('regressor', self.model)
#         ])
#
#     def _text_embeddings(self, X):
#         return DataPreprocessor().generate_embeddings(X.squeeze())
#
#     def train(self, X, y):
#         self.pipeline.fit(X, np.log1p(y))
#
#     def predict(self, X):
#         return np.expm1(self.pipeline.predict(X))
#
#     def optimize(self, X, y):
#         param_grid = {
#             'regressor__max_depth': [3, 5],
#             'regressor__subsample': [0.8],
#             'regressor__colsample_bytree': [0.8]
#         }
#
#         search = RandomizedSearchCV(
#             self.pipeline,
#             param_grid,
#             n_iter=2,
#             scoring='neg_mean_absolute_error',
#             cv=TimeSeriesSplit(n_splits=3),
#             n_jobs=1,
#             error_score='raise'
#         )
#
#         search.fit(X, np.log1p(y))
#         self.pipeline = search.best_estimator_
#
#
# if __name__ == "__main__":
#     # Вчитување и подготовка
#     df = pd.read_csv('kgapp/datasets/datasets_converted.csv', parse_dates=['Date'])
#     df = df.dropna(subset=['Contract', 'Amount'])
#
#     prep = DataPreprocessor()
#     df['Contract'] = df['Contract'].apply(prep.clean_text)
#     df = prep.add_temporal_features(df)
#
#     # Филтрирање
#     q_low, q_high = df['Amount'].quantile(CONFIG["quantile_range"])
#     df = df[(df['Amount'] > q_low) & (df['Amount'] < q_high)]
#
#     # Поделба
#     split_idx = int(len(df) * (1 - CONFIG["test_size"]))
#     train = df.iloc[:split_idx]
#     test = df.iloc[split_idx:]
#
#     # Тренирање
#     predictor = ProcurementPredictor()
#     try:
#         predictor.optimize(train[['Contract', 'Institution', 'Supplier', 'Year', 'Quarter', 'Days']], train['Amount'])
#     except Exception as e:
#         print(f"Грешка при оптимизација: {str(e)}")
#         predictor.train(train[['Contract', 'Institution', 'Supplier', 'Year', 'Quarter', 'Days']], train['Amount'])
#
#     # Евалуација
#     if not test.empty:
#         preds = predictor.predict(test[['Contract', 'Institution', 'Supplier', 'Year', 'Quarter', 'Days']])
#         print(f"MAE: {mean_absolute_error(test['Amount'], preds):,.2f}")
#         print(f"MAPE: {mean_absolute_percentage_error(test['Amount'], preds) * 100:.2f}%")
#         print(f"R²: {r2_score(test['Amount'], preds):.2f}")
#     else:
#         print("Нема податоци за тестирање")
#
#     # Зачувување
#     joblib.dump(predictor, 'procurement_model.pkl')
#
#     # Пример
#     example = pd.DataFrame([{
#         'Contract': 'Набавка на дигитална здравствена опрема',
#         'Institution': 'МИНИСТЕРСТВО ЗА ЗДРАВСТВО',
#         'Supplier': 'ТЕХНОЛОГИИ ДОО',
#         'Year': 2024,
#         'Quarter': 1,
#         'Days': 30
#     }])
#
#     print(f"\nПримерок предвидување: {predictor.predict(example)[0]:,.2f}")
