import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Sample data: Replace this with your actual sales data
# Data should have a date column and a sales column
data = {
    'date': pd.date_range(start='2022-01-01', periods=100, freq='D'),
    'sales': np.random.randint(100, 500, size=100)  # Random sales data
}

# Create a DataFrame
df = pd.DataFrame(data)
df.set_index('date', inplace=True)

# Plot the sales data
df.plot(figsize=(10, 6))
plt.title('Sales Data')
plt.show()

# Split data into training and testing sets
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

# Fit ARIMA model
model = ARIMA(train['sales'], order=(5, 1, 0))  # (p, d, q) parameters
model_fit = model.fit()

# Forecast future sales
forecast = model_fit.forecast(steps=len(test))

# Plot the forecasted sales
plt.figure(figsize=(10, 6))
plt.plot(train.index, train['sales'], label='Training Data')
plt.plot(test.index, test['sales'], label='Actual Sales')
plt.plot(test.index, forecast, label='Forecasted Sales', color='red')
plt.title('Sales Forecasting')
plt.legend()
plt.show()

# Evaluate the model
mse = mean_squared_error(test['sales'], forecast)
print(f'Mean Squared Error: {mse}')