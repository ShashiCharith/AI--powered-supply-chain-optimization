import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
import joblib
import io
from utils import parse_date_column, detect_date_column, preprocess_time_series, prepare_features, calculate_correlation_matrix, handle_nat_dates
from lstm_model import LSTMForecaster
from route_utils import RouteOptimizer
import folium
from streamlit_folium import folium_static
import scipy.stats

# Page config
st.set_page_config(
    page_title="Supply Chain Optimization Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("📊 AI-Powered Supply Chain Optimization Dashboard")

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'test_data' not in st.session_state:
    st.session_state.test_data = None
if 'model_type' not in st.session_state:
    st.session_state.model_type = None
if 'route_data' not in st.session_state:
    st.session_state.route_data = None
if 'selected_product' not in st.session_state:
    st.session_state.selected_product = None

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Product Demand Forecasting", "Route Optimization"])

if page == "Product Demand Forecasting":
    st.header("📈 Product Demand Forecasting")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your product demand dataset (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.data = df
            
            # Display basic info
            st.subheader("📊 Dataset Overview")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Dataset Shape:", df.shape)
                st.write("Columns:", list(df.columns))
            
            with col2:
                st.write("Sample Data:")
                st.dataframe(df.head())
            
            # Product selection
            product_col = None
            if 'Product' in df.columns:
                product_col = 'Product'
            elif 'Product_Name' in df.columns:
                product_col = 'Product_Name'
            
            if product_col:
                unique_products = df[product_col].unique()
                st.subheader("🔍 Product Selection")
                selected_product = st.selectbox("Select a Product", unique_products)
                st.session_state.selected_product = selected_product
                
                # Filter data for selected product
                product_df = df[df[product_col] == selected_product].copy()
                
                # Date column detection and parsing
                date_cols = detect_date_column(product_df)
                if date_cols:
                    date_col = st.selectbox("Select Date Column", date_cols)
                    try:
                        product_df = parse_date_column(product_df, date_col)
                        product_df = handle_nat_dates(product_df, date_col, strategy='forward')
                        st.success(f"Successfully parsed dates in column: {date_col}")
                    except Exception as e:
                        st.error(f"Error parsing dates: {str(e)}")
                        st.info("Please check if the selected column contains valid dates")
                        st.stop()
                else:
                    st.warning("No date columns detected. Please ensure your dataset has a date column.")
                    st.stop()
                
                # Feature engineering
                st.subheader("⚙️ Feature Engineering")
                
                # Extract temporal features
                product_df['Month'] = product_df[date_col].dt.month
                product_df['Week'] = product_df[date_col].dt.isocalendar().week
                product_df['DayOfWeek'] = product_df[date_col].dt.dayofweek
                product_df['Quarter'] = product_df[date_col].dt.quarter
                
                # Handle categorical variables
                categorical_cols = ['Store_Name', 'Product_ID']
                for col in categorical_cols:
                    if col in product_df.columns:
                        # Convert to category type
                        product_df[col] = product_df[col].astype('category')
                
                # Normalize numeric features
                numeric_cols = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
                scaler = StandardScaler()
                for col in numeric_cols:
                    if col in product_df.columns:
                        product_df[f'{col}_Normalized'] = scaler.fit_transform(product_df[[col]])
                
                # Display engineered features
                st.write("Engineered Features Preview:")
                st.dataframe(product_df.head())
                
                # Correlation Matrix
                st.subheader("📊 Correlation Analysis")
                numeric_features = product_df.select_dtypes(include=[np.number]).columns
                if len(numeric_features) > 1:
                    corr_matrix = calculate_correlation_matrix(product_df, numeric_features)
                    st.plotly_chart(corr_matrix, use_container_width=True)
                else:
                    st.info("Not enough numeric features for correlation analysis")
                
                # Model selection
                st.subheader("🤖 Model Selection")
                model_type = st.selectbox(
                    "Select Model Type",
                    ["XGBoost", "ARIMA", "LSTM"]
                )
                st.session_state.model_type = model_type
                
                # Prediction period selection
                forecast_period = st.slider("Select Forecast Period (weeks)", 4, 12, 8)
                
                if model_type == "XGBoost":
                    # XGBoost parameters
                    n_estimators = st.slider("Number of Estimators", 50, 500, 100)
                    max_depth = st.slider("Max Depth", 3, 10, 5)
                    learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1)
                    
                    if st.button("Train Model"):
                        try:
                            # Prepare features
                            feature_cols = [col for col in product_df.columns 
                                          if col not in [date_col, product_col, 'Weekly_Sales', 'Demand']]
                            
                            # Separate numeric and categorical features
                            numeric_features = product_df[feature_cols].select_dtypes(include=['int64', 'float64']).columns
                            categorical_features = product_df[feature_cols].select_dtypes(include=['category']).columns
                            
                            # Prepare X (features)
                            X_numeric = product_df[numeric_features]
                            X_categorical = pd.get_dummies(product_df[categorical_features], prefix=categorical_features)
                            X = pd.concat([X_numeric, X_categorical], axis=1)
                            
                            # Prepare y (target)
                            y = product_df['Weekly_Sales'] if 'Weekly_Sales' in product_df.columns else product_df['Demand']
                            
                            # Split the data
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                            
                            # Train model with categorical features enabled
                            model = xgb.XGBRegressor(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                learning_rate=learning_rate,
                                enable_categorical=True
                            )
                            model.fit(X_train, y_train)
                            
                            # Make predictions
                            y_pred = model.predict(X_test)
                            
                            # Calculate metrics
                            mse = mean_squared_error(y_test, y_pred)
                            rmse = np.sqrt(mse)
                            mae = mean_absolute_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)
                            
                            # Display metrics
                            col1, col2, col3 = st.columns(3)
                            col1.metric("RMSE", f"{rmse:.2f}")
                            col2.metric("MAE", f"{mae:.2f}")
                            col3.metric("R²", f"{r2:.2f}")
                            
                            # Feature importance
                            importance = pd.DataFrame({
                                'feature': X.columns,
                                'importance': model.feature_importances_
                            }).sort_values('importance', ascending=False)
                            
                            fig = px.bar(importance, x='feature', y='importance', title="Feature Importance")
                            st.plotly_chart(fig, use_container_width=True, key="feature_importance")
                            
                            # Save model and data
                            st.session_state.model = model
                            st.session_state.predictions = y_pred
                            st.session_state.test_data = y_test
                            
                            # Model Performance Metrics
                            st.subheader("📊 Model Performance Metrics")
                            
                            # Calculate additional metrics
                            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                            
                            # Display metrics in a grid
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
                                st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")
                                st.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
                            with col2:
                                st.metric("Mean Absolute Percentage Error (MAPE)", f"{mape:.2f}%")
                                st.metric("R-squared (R²)", f"{r2:.2f}")
                            
                            # Create visualization of actual vs predicted values
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                y=y_test,
                                name='Actual Values',
                                mode='lines+markers'
                            ))
                            fig.add_trace(go.Scatter(
                                y=y_pred,
                                name='Predicted Values',
                                mode='lines+markers'
                            ))
                            fig.update_layout(
                                title='Actual vs Predicted Values',
                                xaxis_title='Time Period',
                                yaxis_title='Value',
                                showlegend=True
                            )
                            st.plotly_chart(fig, use_container_width=True, key="actual_vs_predicted")
                            
                            # Residuals plot
                            residuals = y_test - y_pred
                            fig_res = go.Figure()
                            fig_res.add_trace(go.Scatter(
                                y=residuals,
                                name='Residuals',
                                mode='lines+markers'
                            ))
                            fig_res.add_hline(y=0, line_dash="dash", line_color="red")
                            fig_res.update_layout(
                                title='Residuals Plot',
                                xaxis_title='Time Period',
                                yaxis_title='Residual Value',
                                showlegend=True
                            )
                            st.plotly_chart(fig_res, use_container_width=True, key="residuals_plot")
                            
                            # Statistical analysis
                            st.write("### Statistical Analysis")
                            
                            # Basic statistics of residuals
                            residual_stats = pd.DataFrame({
                                'Metric': ['Mean', 'Standard Deviation', 'Min', 'Max'],
                                'Value': [
                                    np.mean(residuals),
                                    np.std(residuals),
                                    np.min(residuals),
                                    np.max(residuals)
                                ]
                            })
                            st.dataframe(residual_stats)
                            
                            # Distribution plot of residuals
                            fig_dist = go.Figure()
                            fig_dist.add_trace(go.Histogram(
                                x=residuals,
                                name='Residuals Distribution',
                                nbinsx=30
                            ))
                            fig_dist.update_layout(
                                title='Distribution of Residuals',
                                xaxis_title='Residual Value',
                                yaxis_title='Frequency',
                                showlegend=True
                            )
                            st.plotly_chart(fig_dist, use_container_width=True, key="residuals_distribution")
                            
                            # Export functionality
                            st.subheader("📥 Export Results")
                            try:
                                # Calculate safety stock and reorder point
                                avg_weekly_demand = np.mean(y_pred)
                                std_dev = np.std(y_pred)
                                safety_stock = 1.96 * std_dev  # 95% service level
                                lead_time = 2  # weeks
                                reorder_point = (avg_weekly_demand * lead_time) + safety_stock
                                
                                # Create forecast dates
                                last_date = product_df[date_col].iloc[-1]
                                forecast_dates = pd.date_range(start=last_date, periods=forecast_period+1)[1:]
                                
                                # Generate future predictions
                                future_features = X.iloc[-forecast_period:].copy()  # Use last forecast_period rows as template
                                future_predictions = model.predict(future_features)
                                
                                # Validate array lengths
                                if len(forecast_dates) != len(future_predictions):
                                    raise ValueError(f"Mismatch in lengths: forecast_dates ({len(forecast_dates)}) != predictions ({len(future_predictions)})")
                                
                                # Create results DataFrame
                                results_df = pd.DataFrame({
                                    'Date': forecast_dates,
                                    'Forecasted_Demand': future_predictions,
                                    'Safety_Stock': [safety_stock] * len(forecast_dates),
                                    'Reorder_Point': [reorder_point] * len(forecast_dates)
                                })
                                
                                # Export to CSV
                                csv = results_df.to_csv(index=False)
                                st.download_button(
                                    label="Download Forecast Results as CSV",
                                    data=csv,
                                    file_name="demand_forecast.csv",
                                    mime="text/csv"
                                )
                                
                                # Save model
                                model_bytes = io.BytesIO()
                                joblib.dump(st.session_state.model, model_bytes)
                                st.download_button(
                                    label="Download Model",
                                    data=model_bytes.getvalue(),
                                    file_name="demand_forecast_model.joblib",
                                    mime="application/octet-stream"
                                )
                            except ValueError as ve:
                                st.error(f"Data validation error: {str(ve)}")
                                st.info("Please ensure all data arrays have consistent lengths")
                            except Exception as e:
                                st.error(f"Error exporting results: {str(e)}")
                                st.info("Please ensure all data arrays have consistent lengths")
                            
                        except Exception as e:
                            st.error(f"Error in XGBoost modeling: {str(e)}")
                            st.info("Please check if your data is properly formatted and contains valid numeric values")
                    
                elif model_type == "ARIMA":
                    if date_cols:
                        # ARIMA parameters
                        p = st.slider("AR Order (p)", 0, 5, 1)
                        d = st.slider("Difference Order (d)", 0, 2, 1)
                        q = st.slider("MA Order (q)", 0, 5, 1)
                        
                        if st.button("Train Model"):
                            try:
                                # Prepare data
                                ts_data = product_df.set_index(date_col)['Weekly_Sales']
                                
                                # Train model
                                model = ARIMA(ts_data, order=(p, d, q))
                                model_fit = model.fit()
                                
                                # Make predictions
                                forecast = model_fit.forecast(steps=forecast_period)
                                
                                # Plot results
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(x=ts_data.index, y=ts_data, name="Actual"))
                                fig.add_trace(go.Scatter(x=pd.date_range(start=ts_data.index[-1], periods=forecast_period+1)[1:],
                                                       y=forecast, name="Forecast"))
                                fig.update_layout(title="ARIMA Forecast")
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Save model and data
                                st.session_state.model = model_fit
                                st.session_state.predictions = forecast
                                st.session_state.test_data = ts_data
                                
                            except Exception as e:
                                st.error(f"Error in ARIMA modeling: {str(e)}")
                                st.info("Please check if your time series data is properly formatted")
                
                elif model_type == "LSTM":
                    if date_cols:
                        # LSTM parameters
                        sequence_length = st.slider("Sequence Length", 5, 30, 10)
                        epochs = st.slider("Number of Epochs", 10, 200, 50)
                        batch_size = st.slider("Batch Size", 16, 128, 32)
                        
                        if st.button("Train Model"):
                            try:
                                # Prepare data
                                ts_data = product_df.set_index(date_col)['Weekly_Sales']
                                
                                # Check if we have enough data
                                if len(ts_data) < sequence_length:
                                    st.error(f"Not enough data points. Need at least {sequence_length} points, but got {len(ts_data)}")
                                    st.stop()
                                
                                # Convert to numpy array
                                ts_data = ts_data.values
                                
                                # Initialize and train LSTM model
                                lstm_model = LSTMForecaster(sequence_length=sequence_length)
                                
                                with st.spinner('Training LSTM model... This may take a few minutes.'):
                                    history = lstm_model.fit(ts_data, epochs=epochs, batch_size=batch_size)
                                
                                # Make predictions
                                predictions = lstm_model.forecast(ts_data, steps=forecast_period)
                                
                                # Plot results
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(x=product_df[date_col], y=product_df['Weekly_Sales'], name="Actual"))
                                fig.add_trace(go.Scatter(x=pd.date_range(start=product_df[date_col].iloc[-1], periods=forecast_period+1)[1:],
                                                       y=predictions, name="Forecast"))
                                fig.update_layout(title="LSTM Forecast")
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Save model and data
                                st.session_state.model = lstm_model
                                st.session_state.predictions = predictions
                                st.session_state.test_data = ts_data
                                
                            except Exception as e:
                                st.error(f"Error in LSTM modeling: {str(e)}")
                                st.info("Please check if your time series data is properly formatted")
                
                # Supply Chain Recommendations
                if st.session_state.model is not None and st.session_state.predictions is not None:
                    st.subheader("📦 Supply Chain Recommendations")
                    
                    # Calculate average weekly demand
                    avg_weekly_demand = np.mean(st.session_state.predictions)
                    
                    # Calculate safety stock (assuming 95% service level)
                    std_dev = np.std(st.session_state.predictions)
                    safety_stock = 1.96 * std_dev
                    
                    # Calculate reorder point
                    lead_time = st.number_input("Enter Lead Time (weeks)", min_value=1, max_value=12, value=2)
                    reorder_point = (avg_weekly_demand * lead_time) + safety_stock
                    
                    # Display recommendations
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Average Weekly Demand", f"{avg_weekly_demand:.2f} units")
                    col2.metric("Safety Stock", f"{safety_stock:.2f} units")
                    col3.metric("Reorder Point", f"{reorder_point:.2f} units")
                    
                    # Additional recommendations
                    st.write("### 📋 Recommendations:")
                    st.write("1. **Inventory Management**:")
                    st.write(f"   - Maintain a minimum safety stock of {safety_stock:.2f} units")
                    st.write(f"   - Reorder when inventory reaches {reorder_point:.2f} units")
                    
                    st.write("2. **Production Planning**:")
                    st.write(f"   - Plan for average weekly production of {avg_weekly_demand:.2f} units")
                    st.write(f"   - Consider seasonal variations in demand")
                    
                    st.write("3. **Logistics Planning**:")
                    st.write(f"   - Schedule deliveries based on {lead_time}-week lead time")
                    st.write("   - Consider increasing lead time buffer during peak seasons")
                    
                    # Export functionality
                    st.subheader("📥 Export Results")
                    try:
                        results_df = pd.DataFrame({
                            'Date': pd.date_range(start=product_df[date_col].iloc[-1], periods=forecast_period+1)[1:],
                            'Forecasted_Demand': st.session_state.predictions,
                            'Safety_Stock': safety_stock,
                            'Reorder_Point': reorder_point
                        })
                        
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Forecast Results as CSV",
                            data=csv,
                            file_name="demand_forecast.csv",
                            mime="text/csv"
                        )
                        
                        # Save model
                        model_bytes = io.BytesIO()
                        joblib.dump(st.session_state.model, model_bytes)
                        st.download_button(
                            label="Download Model",
                            data=model_bytes.getvalue(),
                            file_name="demand_forecast_model.joblib",
                            mime="application/octet-stream"
                        )
                    except Exception as e:
                        st.error(f"Error exporting results: {str(e)}")
            else:
                st.error("No 'Product' or 'Product_Name' column found in the dataset. Please ensure your dataset includes either a 'Product' or 'Product_Name' column.")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("👆 Please upload a CSV file to begin analysis")

elif page == "Route Optimization":
    st.header("🚚 Route Optimization")
    
    # File upload for delivery points
    uploaded_file = st.file_uploader("Upload delivery points (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.route_data = df
            
            # Display basic info
            st.subheader("📍 Delivery Points Overview")
            st.dataframe(df.head())
            
            # Route optimization parameters
            st.subheader("⚙️ Optimization Parameters")
            col1, col2 = st.columns(2)
            
            with col1:
                weather_impact = st.slider("Weather Impact", 0.5, 2.0, 1.0, 
                                         help="Higher values indicate worse weather conditions")
                fuel_price = st.number_input("Fuel Price (₹/L)", 50.0, 150.0, 100.0)
            
            with col2:
                traffic_weight = st.slider("Traffic Weight", 0.5, 2.0, 1.0,
                                         help="Higher values indicate heavier traffic")
                priority_weight = st.slider("Priority Weight", 0.5, 2.0, 1.0,
                                          help="Higher values give more importance to delivery priorities")
            
            if st.button("Optimize Route"):
                try:
                    with st.spinner("Optimizing route..."):
                        # Initialize route optimizer
                        optimizer = RouteOptimizer()
                        
                        # Process delivery data
                        locations, city_names, demands, priorities = optimizer.process_delivery_data(df)
                        
                        if not locations:
                            st.error("No valid locations found. Please check your city names.")
                            st.stop()
                        
                        # Optimize route
                        route = optimizer.optimize_route(
                            locations=locations,
                            demands=demands,
                            priorities=priorities,
                            weather_impact=weather_impact,
                            traffic_weight=traffic_weight
                        )
                        
                        # Calculate metrics
                        metrics = optimizer.calculate_metrics(
                            route=route,
                            locations=locations,
                            demands=demands,
                            fuel_price=fuel_price
                        )
                        
                        # Create map
                        m = optimizer.create_map(route, locations, city_names)
                        
                        # Display results
                        st.subheader("🗺️ Optimized Route")
                        folium_static(m)
                        
                        # Display metrics
                        st.subheader("📊 Route Metrics")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Total Distance", f"{metrics['total_distance']} km")
                        col2.metric("Estimated Time", f"{metrics['estimated_time']} hours")
                        col3.metric("Fuel Cost", f"₹{metrics['fuel_cost']}")
                        if metrics['total_demand'] is not None:
                            col4.metric("Total Demand", f"{metrics['total_demand']} units")
                        
                        # Display route sequence
                        st.subheader("📍 Delivery Sequence")
                        route_df = pd.DataFrame({
                            'Stop': range(1, len(route) + 1),
                            'City': [city_names[i] for i in route],
                            'Demand': [demands[i] for i in route] if demands else None,
                            'Priority': [priorities[i] for i in route] if priorities else None
                        })
                        st.dataframe(route_df)
                        
                except Exception as e:
                    st.error(f"Error in route optimization: {str(e)}")
                    st.info("Please check if your data is properly formatted")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("👆 Please upload a CSV file to begin analysis")
    else:
        st.info("👆 Please upload a CSV file with delivery points") 