import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')
os.makedirs('charts', exist_ok=True)

df = pd.read_csv('train.csv', encoding='latin1')
df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y', errors='coerce')
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month

monthly_sales = df.groupby(df['Order Date'].dt.to_period('M'))['Sales'].sum().reset_index()
monthly_sales['Order Date'] = monthly_sales['Order Date'].dt.to_timestamp()

# Task 2
monthly_sales.set_index('Order Date', inplace=True)
plt.figure(figsize=(12, 4))
plt.plot(monthly_sales.index, monthly_sales['Sales'])
plt.title('Overall Monthly Sales Trend')
plt.savefig('charts/monthly_trend.png')
plt.close()

decomp = seasonal_decompose(monthly_sales['Sales'], model='additive', period=12)
fig = decomp.plot()
fig.set_size_inches(12, 8)
plt.savefig('charts/decomposition.png')
plt.close()

# Task 3
train = monthly_sales['Sales'].iloc[:-3]
test = monthly_sales['Sales'].iloc[-3:]

sarima_model = sm.tsa.statespace.SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,0,12))
sarima_fit = sarima_model.fit(disp=False)
sarima_forecast = sarima_fit.get_forecast(steps=3)
sarima_mean = sarima_forecast.predicted_mean
sarima_conf = sarima_forecast.conf_int()

plt.figure(figsize=(10,4))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test Actual')
plt.plot(sarima_mean.index, sarima_mean, label='SARIMA Forecast')
plt.fill_between(sarima_mean.index, sarima_conf.iloc[:,0], sarima_conf.iloc[:,1], alpha=0.1)
plt.title('SARIMA Forecast')
plt.legend()
plt.savefig('charts/sarima_forecast.png')
plt.close()

prophet_df = monthly_sales.reset_index()[['Order Date', 'Sales']].rename(columns={'Order Date': 'ds', 'Sales': 'y'})
m = Prophet(yearly_seasonality=True)
m.fit(prophet_df.iloc[:-3])
future = m.make_future_dataframe(periods=3, freq='MS')
forecast = m.predict(future)

fig1 = m.plot(forecast)
plt.title('Prophet Forecast')
plt.savefig('charts/prophet_forecast.png')
plt.close()
fig2 = m.plot_components(forecast)
plt.savefig('charts/prophet_components.png')
plt.close()

ml_df = monthly_sales.reset_index()[['Order Date', 'Sales']]
for i in range(1, 4):
    ml_df[f'Lag{i}'] = ml_df['Sales'].shift(i)
ml_df['Rolling3'] = ml_df['Sales'].shift(1).rolling(window=3).mean()
ml_df['Month'] = ml_df['Order Date'].dt.month
ml_df['Quarter'] = ml_df['Order Date'].dt.quarter
ml_df.dropna(inplace=True)

X = ml_df.drop(['Order Date', 'Sales'], axis=1)
y = ml_df['Sales']
train_x, test_x = X.iloc[:-3], X.iloc[-3:]
train_y, test_y = y.iloc[:-3], y.iloc[-3:]

xgb = XGBRegressor(n_estimators=100, random_state=42)
xgb.fit(train_x, train_y)
xgb_pred = xgb.predict(test_x)

plt.figure(figsize=(10,4))
plt.plot(ml_df['Order Date'].iloc[:-3], train_y, label='Train')
plt.plot(ml_df['Order Date'].iloc[-3:], test_y, label='Test Actual')
plt.plot(ml_df['Order Date'].iloc[-3:], xgb_pred, label='XGBoost')
plt.title('XGBoost Forecast')
plt.legend()
plt.savefig('charts/xgb_forecast.png')
plt.close()

# Task 4
segments = {
    'Furniture': df[df['Category'] == 'Furniture'],
    'Technology': df[df['Category'] == 'Technology'],
    'Office Supplies': df[df['Category'] == 'Office Supplies'],
    'West': df[df['Region'] == 'West'],
    'East': df[df['Region'] == 'East']
}
plt.figure(figsize=(12, 6))
for name, seg_df in segments.items():
    s = seg_df.groupby(seg_df['Order Date'].dt.to_period('M'))['Sales'].sum().reset_index()
    s['ds'] = s['Order Date'].dt.to_timestamp()
    s = s[['ds', 'Sales']].rename(columns={'Sales': 'y'})
    m = Prophet(yearly_seasonality=True)
    m.fit(s)
    pred = m.predict(m.make_future_dataframe(periods=3, freq='MS'))
    plt.plot(pred['ds'], pred['yhat'], label=name)

plt.title('3-Month Forecast Across Key Segments (Prophet)')
plt.legend()
plt.savefig('charts/segments.png')
plt.close()

# Task 5
weekly_sales = df.groupby(df['Order Date'].dt.to_period('W'))['Sales'].sum().reset_index()
weekly_sales['Order Date'] = weekly_sales['Order Date'].dt.to_timestamp()
weekly_sales.set_index('Order Date', inplace=True)

iso = IsolationForest(contamination=0.05, random_state=42)
weekly_sales['Anomaly_ISO'] = iso.fit_predict(weekly_sales[['Sales']])
mean = weekly_sales['Sales'].rolling(4).mean()
std = weekly_sales['Sales'].rolling(4).std()
weekly_sales['Z_Score'] = (weekly_sales['Sales'] - mean) / std
weekly_sales['Anomaly_Z'] = weekly_sales['Z_Score'].apply(lambda x: -1 if pd.notnull(x) and abs(x)>2 else 1)

iso_anom = weekly_sales[weekly_sales['Anomaly_ISO']==-1]
z_anom = weekly_sales[weekly_sales['Anomaly_Z']==-1]

plt.figure(figsize=(12,4))
plt.plot(weekly_sales.index, weekly_sales['Sales'], label='Sales Trend')
plt.scatter(iso_anom.index, iso_anom['Sales'], color='red', s=60, label='Iso Forest Anomaly')
plt.scatter(z_anom.index, z_anom['Sales'], color='black', marker='x', s=100, label='Z-Score Anomaly')
plt.legend()
plt.title('Anomalies in Weekly Sales')
plt.savefig('charts/anomalies.png')
plt.close()

# Task 6
prod = df.groupby('Sub-Category').agg(
    Total_Sales=('Sales', 'sum'),
    Volatility=('Sales', 'std'),
    Num_Orders=('Order ID', 'nunique')
)
prod['Volatility'] = prod['Volatility'].fillna(0)
prod['Average_Order'] = prod['Total_Sales'] / prod['Num_Orders']

scaler = StandardScaler()
scaled = scaler.fit_transform(prod)

kmn = KMeans(n_clusters=4, random_state=42).fit(scaled)
prod['Cluster'] = kmn.labels_

pca = PCA(n_components=2)
reduced = pca.fit_transform(scaled)

plt.figure(figsize=(10,6))
sns.scatterplot(x=reduced[:,0], y=reduced[:,1], hue=prod['Cluster'], palette='Set1', legend='full', s=100)
for i, name in enumerate(prod.index):
    plt.annotate(name, (reduced[i,0]+0.1, reduced[i,1]+0.1), fontsize=9)
plt.title('PCA of Product Demand Segments')
plt.savefig('charts/clusters.png')
plt.close()

cluster_names = {
    0: 'High Volume, Volatile', 
    1: 'Stable Volume, Low Volatility', 
    2: 'High Average Order Value', 
    3: 'Hyper-Growth Demand'
}
prod['Cluster_Label'] = prod['Cluster'].map(cluster_names)
prod.to_csv('clusters.csv')
print("All charts generated and saved successfully!")
