import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import os

st.set_page_config(layout="wide", page_title="Dashboard")
st.title("End-to-End Sales Forecasting & Demand Intelligence System")

@st.cache_data
def load_data():
    df = pd.read_csv('train.csv', encoding='latin1')
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y', errors='coerce')
    df['Year'] = df['Order Date'].dt.year
    df['Month'] = df['Order Date'].dt.month
    return df

df = load_data()

tabs = st.tabs(["Sales Overview", "Forecast Explorer", "Anomaly Report", "Demand Segments"])

with tabs[0]:
    st.header("Sales Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Total Sales by Year")
        st.bar_chart(df.groupby('Year')['Sales'].sum())
    with col2:
        st.subheader("Monthly Sales Trend")
        monthly = df.groupby(df['Order Date'].dt.to_period('M'))['Sales'].sum().reset_index()
        monthly['Order Date'] = monthly['Order Date'].dt.to_timestamp()
        st.line_chart(monthly.set_index('Order Date')['Sales'])

with tabs[1]:
    st.header("Forecast Explorer (Powered by Prophet)")
    col1, col2 = st.columns([1, 3])
    with col1:
        seg_type = st.selectbox("Segment Type", ["Category", "Region"])
        val = st.selectbox("Segment Filter", df[seg_type].unique())
        horizon = st.slider("Forecast Horizon (Months)", 1, 3, 3)
    
    with col2:
        sub = df[df[seg_type] == val].groupby(df['Order Date'].dt.to_period('M'))['Sales'].sum().reset_index()
        sub['ds'] = sub['Order Date'].dt.to_timestamp()
        sub = sub[['ds', 'Sales']].rename(columns={'Sales': 'y'})
        
        m = Prophet(yearly_seasonality=True)
        m.fit(sub.iloc[:-3]) # Train excluding last 3 for evaluation
        test_actual = sub['y'].iloc[-3:].values
        
        future = m.make_future_dataframe(periods=horizon+3, freq='MS')
        pred = m.predict(future)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        m.plot(pred, ax=ax)
        ax.set_title(f'Forecast for {val} ({seg_type})')
        st.pyplot(fig)
        
        preds_test = pred['yhat'].iloc[-3-horizon:-horizon].values
        mae = mean_absolute_error(test_actual, preds_test)
        rmse = np.sqrt(mean_squared_error(test_actual, preds_test))
        st.write(f"**Model Performance on {val}:** MAE = {round(mae,2)} | RMSE = {round(rmse,2)}")

with tabs[2]:
    st.header("Anomaly Detection Report")
    if os.path.exists('charts/anomalies.png'):
        st.image('charts/anomalies.png', width=800)
    
    w = df.groupby(df['Order Date'].dt.to_period('W'))['Sales'].sum().reset_index()
    w['Order Date'] = w['Order Date'].dt.to_timestamp()
    iso = IsolationForest(contamination=0.05, random_state=42)
    w['Anomaly'] = iso.fit_predict(w[['Sales']])
    st.subheader("Anomalous Dates & Values (Isolation Forest)")
    st.dataframe(w[w['Anomaly'] == -1][['Order Date', 'Sales', 'Anomaly']])

with tabs[3]:
    st.header("Product Demand Segmentation")
    if os.path.exists('charts/clusters.png'):
        st.image('charts/clusters.png', width=700)
    
    if os.path.exists('clusters.csv'):
        c = pd.read_csv('clusters.csv')
        st.subheader("Sub-Category Clusters")
        st.dataframe(c[['Sub-Category', 'Cluster_Label', 'Total_Sales', 'Average_Order']])
