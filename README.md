# AI-Powered Supply Chain Optimization Dashboard

A comprehensive web-based dashboard for supply chain optimization using machine learning and data analytics.

## Features

- 📊 Interactive data visualization
- 🤖 Multiple ML model support (ARIMA, XGBoost)
- 📈 Time series forecasting
- 🔍 Outlier detection
- 📱 Responsive design
- 📥 CSV/Excel data import/export

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the dashboard:
```bash
streamlit run app.py
```

## Usage

1. Upload your supply chain dataset (CSV format)
2. Select features and target variables
3. Choose and configure your preferred ML model
4. View forecasts and optimization suggestions
5. Export results as needed

## Data Requirements

Your CSV file should contain at least:
- Date column
- Product identifiers
- Demand/Sales data
- Optional: Inventory levels, costs, lead times

## License

MIT License

# Supply Chain Optimization Dashboard

This dashboard provides AI-powered supply chain optimization features including demand forecasting and route optimization.

## CSV File Format Requirements

### For Demand Forecasting

The CSV file should contain the following columns:

#### Required Columns:
1. **Date/Time Column** (one of the following):
   - `date`: Full date in any of these formats:
     - YYYY-MM-DD (e.g., 2024-01-01)
     - DD-MM-YYYY (e.g., 01-01-2024)
     - MM-DD-YYYY (e.g., 01-01-2024)
     - Month DD YYYY (e.g., Jan 01 2024)
     - YYYY (e.g., 2024) - will be converted to January 1st of that year

2. **Product Information** (one of the following):
   - `Product`: Product name or identifier
   - `Product_Name`: Alternative column name for product information

3. **Demand/Sales Information** (one of the following):
   - `Weekly_Sales`: Weekly sales figures
   - `Demand`: Weekly demand figures

#### Optional Columns:
1. **Store Information**:
   - `Store_Name`: Name or identifier of the store
   - `Product_ID`: Unique identifier for the product

2. **External Factors**:
   - `Temperature`: Temperature data (if available)
   - `Fuel_Price`: Fuel price data (if available)
   - `CPI`: Consumer Price Index (if available)
   - `Unemployment`: Unemployment rate (if available)

### For Route Optimization

The CSV file should contain the following columns:

#### Required Columns:
1. **Location Information**:
   - `City`: City name or location identifier
   - `Latitude`: Latitude coordinates
   - `Longitude`: Longitude coordinates

#### Optional Columns:
1. **Delivery Information**:
   - `Demand`: Delivery demand or quantity
   - `Priority`: Delivery priority (higher numbers indicate higher priority)

## Example CSV Format

### Demand Forecasting Example:
```csv
date,Product,Weekly_Sales,Store_Name,Temperature,Fuel_Price,CPI,Unemployment
2024-01-01,Product A,100,Store 1,25.5,3.50,120.5,5.2
2024-01-08,Product A,120,Store 1,26.0,3.55,121.0,5.1
```

### Route Optimization Example:
```csv
City,Latitude,Longitude,Demand,Priority
New York,40.7128,-74.0060,100,1
Los Angeles,34.0522,-118.2437,150,2
Chicago,41.8781,-87.6298,80,3
```

## Data Preprocessing

### Target Variable Cleaning
The system automatically handles invalid values in the target variable (Weekly_Sales or Demand) using the following strategies:

1. **Drop Strategy** (default):
   - Removes rows with invalid values (NaN, inf, -inf)
   - Best for maintaining data quality
   - Use when you have enough data points

2. **Mean Strategy**:
   - Replaces invalid values with the mean of valid values
   - Good for maintaining data size
   - Use when missing values are random

3. **Median Strategy**:
   - Replaces invalid values with the median of valid values
   - More robust to outliers than mean
   - Use when data has outliers

4. **Zero Strategy**:
   - Replaces invalid values with 0
   - Use when missing values represent no sales/demand

Example usage in code:
```python
from utils import clean_target_variable

# Clean target variable before model training
y_cleaned, valid_indices = clean_target_variable(y, strategy='drop')
X_cleaned = X[valid_indices]  # Apply same cleaning to features

# Then use X_cleaned and y_cleaned for training
model.fit(X_cleaned, y_cleaned)
```

## Notes

1. **Date Formats**:
   - The system supports multiple date formats
   - For year-only columns, dates will be set to January 1st of that year
   - Timezone information (if present) will be handled automatically

2. **Data Quality**:
   - Missing values will be handled automatically
   - Outliers will be detected and can be reviewed
   - Invalid dates will be reported with examples
   - Invalid target values will be cleaned using the specified strategy

3. **File Requirements**:
   - File must be in CSV format
   - UTF-8 encoding is recommended
   - Maximum file size: 100MB
   - No special characters in column names

4. **Best Practices**:
   - Use consistent date formats within the same column
   - Ensure numeric columns contain only numbers
   - Remove any special characters from column names
   - Include a header row with column names 