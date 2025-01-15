import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

# Load and preprocess data
df = pd.read_csv("./data.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['date'] = df['timestamp'].dt.date
df['hour'] = df['timestamp'].dt.hour
df['month'] = df['timestamp'].dt.month
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['year'] = df['timestamp'].dt.year

# Trend Analysis using Linear Regression
agg_time = df.groupby('timestamp')['watts'].sum().reset_index()
X = np.arange(len(agg_time)).reshape(-1, 1)  # Time index as independent variable
y = agg_time['watts'].values.reshape(-1, 1)

model = LinearRegression()
model.fit(X, y)
trend = model.predict(X)

plt.figure(figsize=(12, 6))
plt.plot(agg_time['timestamp'], agg_time['watts'], label='Actual Consumption', color='blue')
plt.plot(agg_time['timestamp'], trend, label='Trend Line', color='red')
plt.title('Trend Analysis of Energy Consumption')
plt.xlabel('Time')
plt.ylabel('Total Watts')
plt.legend()
plt.grid(True)
plt.show()

# Seasonal Decomposition
agg_time.set_index('timestamp', inplace=True)
result = seasonal_decompose(agg_time['watts'], model='additive', period=24)  # Assuming hourly data
result.plot()
plt.show()

# Clustering Nodes Based on Energy Consumption Patterns
node_avg = df.groupby('node')['watts'].mean().reset_index()
kmeans = KMeans(n_clusters=3, random_state=42)
node_avg['cluster'] = kmeans.fit_predict(node_avg[['watts']])

plt.figure(figsize=(10, 6))
sns.scatterplot(data=node_avg, x='node', y='watts', hue='cluster', palette='viridis', s=100)
plt.title('Clustering of Nodes by Energy Consumption')
plt.xlabel('Node')
plt.ylabel('Average Energy Consumption (Watts)')
plt.legend(title='Cluster')
plt.show()

# Forecasting with ARIMA
time_series = agg_time['watts']
train_size = int(len(time_series) * 0.8)
train, test = time_series[:train_size], time_series[train_size:]

model_arima = ARIMA(train, order=(5, 1, 0))  # Example ARIMA(5, 1, 0)
model_fit = model_arima.fit()

forecast = model_fit.forecast(steps=len(test))
plt.figure(figsize=(12, 6))
plt.plot(time_series.index, time_series, label='Actual', color='blue')
plt.plot(test.index, forecast, label='Forecast', color='red')
plt.title('ARIMA Forecast of Energy Consumption')
plt.xlabel('Time')
plt.ylabel('Watts')
plt.legend()
plt.grid(True)
plt.show()

# Insights from Clustering
print("Cluster Centers:")
print(kmeans.cluster_centers_)

# Summarize seasonal decomposition
print("Seasonal Component Example:")
print(result.seasonal.head())
