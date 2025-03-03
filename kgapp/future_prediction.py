import os
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('kgapp/datasets/datasets_converted.csv', encoding='utf-8')

# Keep only Date and Amount columns
data = data[['Date', 'Amount']]

# Convert Date to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Group by year and sum the amounts
yearly_data = data.groupby(data['Date'].dt.year)['Amount'].sum().reset_index()
yearly_data.columns = ['Year', 'Amount']

# Train ARIMA model
arima_model = ARIMA(yearly_data['Amount'], order=(1,1,1))
arima_results = arima_model.fit()

# Predict for the next 5 years
future_years = pd.DataFrame({'Year': range(yearly_data['Year'].max() + 1, yearly_data['Year'].max() + 6)})
arima_future_predictions = arima_results.forecast(steps=5)

# Create future_trends DataFrame
future_trends = future_years.copy()
future_trends['Predicted_Amount'] = arima_future_predictions

# Save future_trends to CSV
os.makedirs('kgapp/datasets/', exist_ok=True)
future_trends.to_csv('kgapp/datasets/future_trends.csv', index=False)

# Visualize the results
plt.figure(figsize=(12, 6))
plt.plot(yearly_data['Year'], yearly_data['Amount'], label='Историски податоци', marker='o')
plt.plot(future_years['Year'], arima_future_predictions, label='ARIMA предвидување', marker='o')
plt.title('Историски податоци и предвидувања за вкупната сума на набавки')
plt.xlabel('Година')
plt.ylabel('Вкупен износ')
plt.legend()

# Save the chart
os.makedirs('kgapp/data/', exist_ok=True)
plt.savefig('kgapp/data/trend_analysis.png')
plt.close()

print("ARIMA predictions for the next 5 years:")
print(arima_future_predictions)
print("Trend analysis completed and results saved.")
