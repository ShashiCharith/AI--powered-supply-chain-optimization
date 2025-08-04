import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re

def clean_date_string(date_str):
    """Clean date string by removing common issues and standardizing formats."""
    if pd.isna(date_str):
        return date_str
    if isinstance(date_str, str):
        # Remove timezone information (e.g., GMT-0800 (PST))
        date_str = re.sub(r'\s*GMT[+-]\d{4}\s*\([A-Z]+\)', '', date_str)
        
        # Remove any non-date characters at the end
        date_str = re.sub(r'[^\d\s\-/\.:]+$', '', date_str)
        
        # Remove any extra spaces
        date_str = date_str.strip()
        
        # Replace multiple spaces with single space
        date_str = re.sub(r'\s+', ' ', date_str)
        
        # Handle common date format issues
        # Convert MM-DD-YYYY to YYYY-MM-DD
        date_str = re.sub(r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})', r'\3-\1-\2', date_str)
        
        # Convert DD-MM-YYYY to YYYY-MM-DD
        date_str = re.sub(r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})', r'\3-\2-\1', date_str)
        
        # Handle month names (e.g., "Dec 16 2014")
        month_map = {
            'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
            'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
            'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
        }
        for month, num in month_map.items():
            date_str = re.sub(
                rf'{month}\s+(\d{{1,2}})\s+(\d{{4}})',
                rf'\2-{num}-\1',
                date_str,
                flags=re.IGNORECASE
            )
    return date_str

def parse_date_column(df, date_col):
    """Robustly parse date column with multiple format support."""
    # Make a copy of the dataframe to avoid modifying the original
    df = df.copy()
    
    # Check if the column contains only year values (4-digit numbers)
    # If it's a year column, convert to January 1st of that year
    # Example: 2014 -> 2014-01-01
    if df[date_col].astype(str).str.match(r'^\d{4}$').all():
        df[date_col] = pd.to_datetime(df[date_col].astype(str) + '-01-01')
        return df
    
    # Clean the date column
    df[date_col] = df[date_col].apply(clean_date_string)
    
    # First try with dateutil parser for maximum flexibility
    try:
        from dateutil import parser as dateutil_parser
        df[date_col] = df[date_col].apply(lambda x: dateutil_parser.parse(x) if pd.notna(x) else x)
        if df[date_col].notna().any():
            return df
    except:
        pass
    
    # Common date formats to try
    date_formats = [
        '%Y-%m-%d', '%d-%m-%Y', '%m-%d-%Y', '%Y/%m/%d', '%d/%m/%Y', '%m/%d/%Y',
        '%Y-%m-%d %H:%M:%S', '%d-%m-%Y %H:%M:%S', '%m-%d-%Y %H:%M:%S',
        '%Y/%m/%d %H:%M:%S', '%d/%m/%Y %H:%M:%S', '%m/%d/%Y %H:%M:%S',
        '%Y%m%d', '%d%m%Y', '%m%d%Y',
        '%b %d %Y', '%B %d %Y', '%d %b %Y', '%d %B %Y',
        '%Y %b %d', '%Y %B %d',
        '%b-%d-%Y', '%B-%d-%Y', '%d-%b-%Y', '%d-%B-%Y',
        '%Y-%b-%d', '%Y-%B-%d',
        '%d-%b-%y', '%d-%B-%y', '%b-%d-%y', '%B-%d-%y',
        '%y-%b-%d', '%y-%B-%d',
        '%d/%b/%y', '%d/%B/%y', '%b/%d/%y', '%B/%d/%y',
        '%y/%b/%d', '%y/%B/%d',
        '%d %b %y', '%d %B %y', '%b %d %y', '%B %d %y',
        '%y %b %d', '%y %B %d',
        # Additional formats
        '%a %b %d %Y %H:%M:%S',  # For "Tue Dec 16 2014 12:30:00"
        '%a %b %d %Y',           # For "Tue Dec 16 2014"
        '%Y-%m-%dT%H:%M:%S',     # ISO format
        '%Y-%m-%dT%H:%M:%S.%f'   # ISO format with microseconds
    ]
    
    # Try parsing with different formats
    for date_format in date_formats:
        try:
            df[date_col] = pd.to_datetime(df[date_col], format=date_format, errors='coerce')
            # Check if we successfully parsed any dates
            if df[date_col].notna().any():
                return df
        except:
            continue
    
    # If specific formats fail, try pandas' flexible parser with different options
    parsing_options = [
        {'dayfirst': False, 'yearfirst': False},  # Default
        {'dayfirst': True, 'yearfirst': False},   # European style
        {'dayfirst': False, 'yearfirst': True},   # ISO style
        {'dayfirst': True, 'yearfirst': True}     # Mixed style
    ]
    
    for options in parsing_options:
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce', **options)
            if df[date_col].notna().any():
                return df
        except:
            continue
    
    # If all parsing attempts fail, try to infer the format from the data
    try:
        # Get a sample of non-null dates
        sample_dates = df[date_col].dropna().head()
        if not sample_dates.empty:
            # Try to infer the format from the first valid date
            first_date = str(sample_dates.iloc[0])
            inferred_format = None
            
            # Common patterns
            patterns = {
                r'\d{4}-\d{2}-\d{2}': '%Y-%m-%d',
                r'\d{2}-\d{2}-\d{4}': '%d-%m-%Y',
                r'\d{2}/\d{2}/\d{4}': '%d/%m/%Y',
                r'\d{4}/\d{2}/\d{2}': '%Y/%m/%d',
                r'[A-Za-z]{3}\s+\d{1,2}\s+\d{4}': '%b %d %Y',  # For "Dec 16 2014"
                r'[A-Za-z]{3}\s+\d{1,2}\s+\d{4}\s+\d{2}:\d{2}:\d{2}': '%b %d %Y %H:%M:%S'  # For "Dec 16 2014 12:30:00"
            }
            
            for pattern, date_format in patterns.items():
                if re.match(pattern, first_date):
                    inferred_format = date_format
                    break
            
            if inferred_format:
                df[date_col] = pd.to_datetime(df[date_col], format=inferred_format, errors='coerce')
                if df[date_col].notna().any():
                    return df
    except:
        pass
    
    # If all parsing attempts fail, raise a more informative error
    invalid_dates = df[df[date_col].isna()][date_col].unique()
    error_msg = f"Could not parse dates in column '{date_col}'. "
    if len(invalid_dates) > 0:
        error_msg += f"Example invalid dates: {invalid_dates[:5]}"
    raise ValueError(error_msg)

def detect_date_column(df):
    """Detect potential date columns in the dataframe."""
    date_cols = []
    for col in df.columns:
        # Check if column name suggests it's a date
        if any(term in col.lower() for term in ['date', 'time', 'day', 'month', 'year']):
            try:
                # Try parsing with flexible parser
                pd.to_datetime(df[col], errors='coerce')
                # If we successfully parsed any dates, add to list
                if pd.to_datetime(df[col], errors='coerce').notna().any():
                    date_cols.append(col)
            except:
                continue
    return date_cols[0] if date_cols else None

def preprocess_time_series(df, date_col, target_col, feature_cols=None, freq='D'):
    """Preprocess time series data with resampling and missing value handling."""
    # Convert to datetime using robust parser
    df = parse_date_column(df, date_col)
    
    # Set as index and sort
    df = df.set_index(date_col).sort_index()
    
    # Handle missing values in target and features
    if feature_cols:
        df[feature_cols] = df[feature_cols].fillna(method='ffill').fillna(method='bfill')
    df[target_col] = df[target_col].fillna(method='ffill').fillna(method='bfill')
    
    # Resample to specified frequency
    df_resampled = df.resample(freq).mean()
    
    # Handle any remaining missing values after resampling
    df_resampled = df_resampled.fillna(method='ffill').fillna(method='bfill')
    
    return df_resampled

def detect_outliers(df, columns, contamination=0.1):
    """Detect outliers using Isolation Forest."""
    # Initialize the model
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    
    # Fit and predict
    outliers = pd.DataFrame()
    for col in columns:
        # Reshape for sklearn
        X = df[col].values.reshape(-1, 1)
        
        # Fit and predict
        pred = iso_forest.fit_predict(X)
        
        # Store results
        outliers[f'{col}_outlier'] = pred == -1
    
    return outliers

def plot_outliers(df, date_col, value_col, outliers):
    """Create an interactive plot showing outliers."""
    fig = go.Figure()
    
    # Add normal points
    normal_mask = ~outliers[f'{value_col}_outlier']
    fig.add_trace(go.Scatter(
        x=df.index[normal_mask],
        y=df[value_col][normal_mask],
        mode='markers',
        name='Normal',
        marker=dict(color='blue')
    ))
    
    # Add outlier points
    outlier_mask = outliers[f'{value_col}_outlier']
    fig.add_trace(go.Scatter(
        x=df.index[outlier_mask],
        y=df[value_col][outlier_mask],
        mode='markers',
        name='Outlier',
        marker=dict(color='red', size=10)
    ))
    
    fig.update_layout(
        title=f'Outlier Detection for {value_col}',
        xaxis_title='Date',
        yaxis_title=value_col,
        showlegend=True
    )
    
    return fig

def calculate_correlation_matrix(df, numeric_cols):
    """Calculate and visualize correlation matrix."""
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        labels=dict(color="Correlation"),
        x=corr_matrix.columns,
        y=corr_matrix.columns,
    )
    
    return fig

def clean_target_variable(y, strategy='drop'):
    """
    Clean target variable by handling invalid values (NaN, inf, -inf).
    
    Parameters:
    -----------
    y : pandas.Series or numpy.ndarray
        Target variable to clean
    strategy : str, default 'drop'
        Strategy to handle invalid values:
        - 'drop': Remove rows with invalid values
        - 'mean': Replace invalid values with mean
        - 'median': Replace invalid values with median
        - 'zero': Replace invalid values with 0
    
    Returns:
    --------
    tuple
        (cleaned_target, valid_indices) where:
        - cleaned_target is the cleaned target variable
        - valid_indices are the indices of valid rows
    """
    import numpy as np
    import pandas as pd
    
    # Convert to pandas Series if numpy array
    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    
    # Store original indices
    original_indices = y.index
    
    # Replace inf values with NaN
    y = y.replace([np.inf, -np.inf], np.nan)
    
    # Get valid indices
    valid_indices = y.notna()
    
    if strategy == 'drop':
        # Drop rows with NaN values
        y_cleaned = y.dropna()
    elif strategy == 'mean':
        # Replace NaN with mean
        y_cleaned = y.fillna(y.mean())
    elif strategy == 'median':
        # Replace NaN with median
        y_cleaned = y.fillna(y.median())
    elif strategy == 'zero':
        # Replace NaN with 0
        y_cleaned = y.fillna(0)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Print information about cleaning
    n_invalid = (~valid_indices).sum()
    if n_invalid > 0:
        print(f"Found {n_invalid} invalid values in target variable")
        print(f"Strategy '{strategy}' applied")
        print(f"Final shape: {y_cleaned.shape}")
    
    return y_cleaned, valid_indices

def handle_nat_dates(df, date_col, strategy='forward'):
    """
    Handle NaT (Not a Time) values in date columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    date_col : str
        Name of the date column
    strategy : str, default 'forward'
        Strategy to handle NaT values:
        - 'forward': Forward fill NaT values
        - 'backward': Backward fill NaT values
        - 'drop': Remove rows with NaT values
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with handled NaT values
    """
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Count NaT values
    nat_count = df[date_col].isna().sum()
    if nat_count == 0:
        return df
    
    if strategy == 'forward':
        df[date_col] = df[date_col].fillna(method='ffill')
    elif strategy == 'backward':
        df[date_col] = df[date_col].fillna(method='bfill')
    elif strategy == 'drop':
        df = df.dropna(subset=[date_col])
    
    return df

def prepare_features(df, target_col, date_col=None):
    """Prepare features for model training."""
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove target column
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    # Create feature matrix
    X = df[numeric_cols]
    y = df[target_col]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return pd.DataFrame(X_scaled, columns=numeric_cols), y, scaler 