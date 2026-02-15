import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.preprocessing import (
    load_volume_data, prepare_features, get_feature_columns,
    split_data, get_peak_hours, get_peak_days, get_summary_stats
)
from src.model import (
    get_model, train_model, predict, evaluate_model,
    get_feature_importance, forecast_future
)
from src.visualization import (
    plot_time_series, plot_hourly_pattern, plot_daily_pattern,
    plot_monthly_pattern, plot_predictions, plot_feature_importance,
    plot_residuals, plot_heatmap, plot_forecast
)

st.set_page_config(page_title="EV Charging Demand Prediction", layout="wide")
st.title("EV Charging Demand Prediction System")
st.markdown("Predict EV charging demand using historical data from charging stations.")

st.sidebar.header("Configuration")
data_source = st.sidebar.radio("Select Data Source", ["Sample Data", "Local File Path", "Upload Data"])

def load_sample_data():
    sample_path = "data/volume.csv"
    if os.path.exists(sample_path):
        return load_volume_data(sample_path)
    else:
        st.error("Sample data not found. Please upload your own data or download the sample data.")
        return None

def load_local_file(filepath):
    if os.path.exists(filepath):
        return load_volume_data(filepath)
    else:
        st.error(f"File not found: {filepath}")
        return None

def run_analysis(df):
    st.header("1. Data Overview")
    df_processed = prepare_features(df)
    stats = get_summary_stats(df_processed)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{stats['total_records']:,}")
    col2.metric("Date Range", f"{stats['date_range_start']} to {stats['date_range_end']}")
    col3.metric("Avg Volume", f"{stats['avg_volume']:.2f} kWh")
    col4.metric("Max Volume", f"{stats['max_volume']:.2f} kWh")
    
    st.subheader("Sample Data")
    st.dataframe(df.head(10))
    
    st.header("2. Demand Patterns")
    tab1, tab2, tab3, tab4 = st.tabs(["Time Series", "Hourly", "Daily", "Heatmap"])
    with tab1:
        fig_ts = plot_time_series(df_processed)
        st.plotly_chart(fig_ts, use_container_width=True)
    with tab2:
        fig_hourly = plot_hourly_pattern(df_processed)
        st.plotly_chart(fig_hourly, use_container_width=True)
    with tab3:
        fig_daily = plot_daily_pattern(df_processed)
        st.plotly_chart(fig_daily, use_container_width=True)
    with tab4:
        fig_heatmap = plot_heatmap(df_processed)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    st.header("3. Peak Demand Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Peak Hours")
        peak_hours = get_peak_hours(df_processed)
        for hour, volume in peak_hours.items():
            st.write(f"Hour {hour}:00 - {volume:.2f} kWh avg")
    with col2:
        st.subheader("Peak Days")
        peak_days = get_peak_days(df_processed)
        for day, volume in peak_days.items():
            st.write(f"{day} - {volume:.2f} kWh avg")
    
    st.header("4. Model Training and Prediction")
    train_ratio = st.slider("Training Data Ratio", 0.6, 0.9, 0.8)
    
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            feature_cols = get_feature_columns()
            train_df, test_df = split_data(df_processed, train_ratio)
            X_train = train_df[feature_cols].values
            y_train = train_df['total_volume'].values
            X_test = test_df[feature_cols].values
            y_test = test_df['total_volume'].values
            
            model = get_model()
            model = train_model(model, X_train, y_train)
            
            y_pred_train = predict(model, X_train)
            y_pred_test = predict(model, X_test)
            
            st.session_state['model'] = model
            st.session_state['test_df'] = test_df
            st.session_state['y_test'] = y_test
            st.session_state['y_pred_test'] = y_pred_test
            st.session_state['feature_cols'] = feature_cols
            st.session_state['df_processed'] = df_processed
            
            st.subheader("Model Performance")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Training Metrics")
                train_metrics = evaluate_model(y_train, y_pred_train)
                for metric, value in train_metrics.items():
                    st.metric(metric, value)
            with col2:
                st.write("Testing Metrics")
                test_metrics = evaluate_model(y_test, y_pred_test)
                for metric, value in test_metrics.items():
                    st.metric(metric, value)
            
            st.subheader("Predictions vs Actual")
            fig_pred = plot_predictions(y_test, y_pred_test, test_df['time'].values)
            st.plotly_chart(fig_pred, use_container_width=True)
            
            importance = get_feature_importance(model, feature_cols)
            st.subheader("Feature Importance")
            fig_imp = plot_feature_importance(importance)
            st.plotly_chart(fig_imp, use_container_width=True)
            
            st.subheader("Residuals Distribution")
            fig_res = plot_residuals(y_test, y_pred_test)
            st.plotly_chart(fig_res, use_container_width=True)
    
    st.header("5. Future Demand Forecast")
    if 'model' in st.session_state:
        forecast_hours = st.slider("Forecast Hours", 1, 48, 24)
        if st.button("Generate Forecast"):
            model = st.session_state['model']
            feature_cols = st.session_state['feature_cols']
            df_processed = st.session_state['df_processed']
            last_row = df_processed.iloc[-1].copy()
            forecast = forecast_future(model, last_row, feature_cols, steps=forecast_hours)
            last_time = df_processed['time'].iloc[-1]
            forecast_times = [last_time + timedelta(hours=i+1) for i in range(forecast_hours)]
            st.subheader("Forecasted Demand")
            historical_times = df_processed['time'].tail(100).values
            historical_values = df_processed['total_volume'].tail(100).values
            fig_fc = plot_forecast(historical_times, historical_values, forecast_times, forecast)
            st.plotly_chart(fig_fc, use_container_width=True)
            forecast_df = pd.DataFrame({'Time': forecast_times, 'Predicted Volume (kWh)': forecast})
            st.dataframe(forecast_df)
    else:
        st.info("Train a model first to generate forecasts.")

if data_source == "Sample Data":
    df = load_sample_data()
    if df is not None:
        run_analysis(df)
elif data_source == "Local File Path":
    st.sidebar.markdown("**Tip:** Copy your file to the workspace and enter the path below.")
    local_path = st.sidebar.text_input("File Path", value="data/volume.csv", placeholder="e.g., data/myfile.csv")
    if st.sidebar.button("Load File"):
        df = load_local_file(local_path)
        if df is not None:
            run_analysis(df)
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            run_analysis(df)
        else:
            st.error("CSV must contain a 'time' column.")
    else:
        st.info("Please upload a CSV file with EV charging data.")
        st.markdown("""
        **Expected CSV Format:**
        - First column: `time` (timestamp format: YYYY-MM-DD HH:MM)
        - Remaining columns: Zone IDs with charging volume data
        
        The data should contain hourly or 5-minute interval charging volume data.
        """)
