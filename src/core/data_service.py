import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import datetime
import logging
from typing import Dict, Tuple, Optional, List
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataService:
    def __init__(self, config: Dict):
        """
        Initialize DataService with configuration
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.scaler = MinMaxScaler()
        
    def load_data(self, csv_path: str) -> Optional[pd.DataFrame]:
        """
        Load data from CSV file and validate
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            DataFrame or None if validation fails
        """
        try:
            # Try to load the specified CSV
            df = pd.read_csv(csv_path)
            logger.info(f"Successfully loaded data from {csv_path}")
            return df
        except Exception as e:
            logger.error(f"Failed to load data from {csv_path}: {e}")
            
            # Fallback to another source
            fallback_path = self._get_fallback_path(csv_path)
            if fallback_path:
                try:
                    df = pd.read_csv(fallback_path)
                    logger.info(f"Successfully loaded fallback data from {fallback_path}")
                    return df
                except Exception as fallback_e:
                    logger.error(f"Failed to load fallback data: {fallback_e}")
            
            return None
    
    def _get_fallback_path(self, original_path: str) -> Optional[str]:
        """Get fallback data path if original fails"""
        # In a real system, this would attempt to fetch from Yahoo Finance
        # For this implementation, we'll look for a file with "_fallback" suffix
        base_path = os.path.splitext(original_path)[0]
        fallback_path = f"{base_path}_fallback.csv"
        
        if os.path.exists(fallback_path):
            return fallback_path
        return None

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data:
        - Convert Time to datetime index
        - Validate data
        - Handle missing values and outliers

        Args:
            df: Input DataFrame

        Returns:
            Preprocessed DataFrame
        """
        logger.info("Starting data preprocessing")

        # Convert Time to datetime index
        if 'Time' in df.columns:
            # Explicitly specify format to avoid warnings
            if 'Date' in df.columns:
                # If we have both Date and Time columns, combine them
                df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y-%m-%d %H:%M:%S',
                                                errors='coerce')
                df.set_index('DateTime', inplace=True)
                df.drop(columns=['Date', 'Time'], inplace=True, errors='ignore')
            else:
                # If we only have Time column
                df['Time'] = pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
                df.set_index('Time', inplace=True)

        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.any():
            logger.warning(f"Missing values detected: {missing_values}")
            # Fill missing values with linear interpolation
            df = df.interpolate(method='linear')

        # Check for outliers
        df = self._handle_outliers(df)

        # Filter by liquidity
        if 'Volume' in df.columns:
            df = df[df['Volume'] >= 100]

        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in the data"""
        # Check for invalid OHLC relationships (High < Low)
        invalid_ohlc = (df['High'] < df['Low']).sum()
        if invalid_ohlc > 0:
            logger.warning(f"Found {invalid_ohlc} instances where High < Low")
            # Fix by swapping values
            invalid_idx = df['High'] < df['Low']
            df.loc[invalid_idx, ['High', 'Low']] = df.loc[invalid_idx, ['Low', 'High']].values
        
        # Cap extreme values at 3 standard deviations
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df.columns:
                mean, std = df[col].mean(), df[col].std()
                lower_bound, upper_bound = mean - 3*std, mean + 3*std
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                
                if outliers > 0:
                    logger.warning(f"Capping {outliers} outliers in {col}")
                    df[col] = df[col].clip(lower_bound, upper_bound)
        
        return df
    
    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize all numeric features to 0-1 range"""
        numeric_cols = df.select_dtypes(include=np.number).columns
        df_normalized = df.copy()
        df_normalized[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        
        return df_normalized

    def resample_data(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Resample data to multiple timeframes based on config

        Args:
            df: Input DataFrame with datetime index

        Returns:
            Dictionary with resampled DataFrames
        """
        sample_rate = self.config.get('sample_rate', '15-minute')

        # If the data is already at the requested sample rate
        if sample_rate == '15-minute':
            # No resampling needed
            return {'15-minute': df}

        # Define resampling rules
        resample_rules = {
            '5-minute': '5min',
            '15-minute': '15min',
            '1-hour': '1H'
        }

        resampled = {}

        # Resample OHLC data
        for timeframe, rule in resample_rules.items():
            if timeframe == sample_rate or sample_rate == 'multi':
                resampled[timeframe] = df.resample(rule).agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                })

                # Handle EMAs
                ema_cols = [col for col in df.columns if col.startswith('EMA')]
                if ema_cols:
                    for ema_col in ema_cols:
                        resampled[timeframe][ema_col] = df[ema_col].resample(rule).last()

        return resampled

    def create_regime_aware_split(self, df: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], Dict[str, List[str]]]:
        """
        Create train/val/test splits that are aware of market regimes

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with split DataFrames and regime info
        """
        # Extract splits from config
        train_pct = self.config.get('split', {}).get('train', 0.7)
        val_pct = self.config.get('split', {}).get('val', 0.15)
        test_pct = self.config.get('split', {}).get('test', 0.15)

        # Make a copy of the dataframe to avoid modifying the original
        df_copy = df.copy()

        # Calculate returns for regime clustering
        df_copy['returns'] = df_copy['Close'].pct_change()

        # Use a rolling window to calculate volatility
        df_copy['volatility'] = df_copy['returns'].rolling(window=20).std()

        # Create features for regime clustering - drop rows with NaN values
        df_copy = df_copy.dropna(subset=['returns', 'volatility'])

        # Initialize regime column with zeros
        df_copy['regime'] = 0

        # Only perform clustering if we have enough data
        if len(df_copy) >= 3:  # Need at least 3 points for 3 clusters
            # K-means clustering to identify regimes
            kmeans = KMeans(n_clusters=min(3, len(df_copy)), random_state=42)
            cluster_features = df_copy[['returns', 'volatility']]
            cluster_features_scaled = MinMaxScaler().fit_transform(cluster_features)

            # Fit and predict
            df_copy['regime'] = kmeans.fit_predict(cluster_features_scaled)

        # Get regimes and their counts
        regimes = df_copy['regime'].unique()
        regime_counts = {int(regime): (df_copy['regime'] == regime).sum() for regime in regimes}
        logger.info(f"Identified regimes with counts: {regime_counts}")

        # First do a chronological split
        n = len(df_copy)
        train_end = int(n * train_pct)
        val_end = train_end + int(n * val_pct)

        train_df = df_copy.iloc[:train_end]
        val_df = df_copy.iloc[train_end:val_end]
        test_df = df_copy.iloc[val_end:]

        # Check regime representation
        train_regimes = set(train_df['regime'].unique())
        val_regimes = set(val_df['regime'].unique())
        test_regimes = set(test_df['regime'].unique())

        all_regimes = set(regimes)

        # If any split is missing a regime, adjust splits
        if (train_regimes != all_regimes or
                val_regimes != all_regimes or
                test_regimes != all_regimes):

            logger.warning("Regimes not represented in all splits, adjusting...")

            # Ensure representation by adding samples from each regime
            for regime in all_regimes:
                regime_indices = df_copy[df_copy['regime'] == regime].index.tolist()

                if regime not in train_regimes and len(regime_indices) > 0:
                    sample_idx = np.random.choice(regime_indices, min(100, len(regime_indices)), replace=False)
                    train_df = pd.concat([train_df, df_copy.loc[sample_idx]])

                if regime not in val_regimes and len(regime_indices) > 0:
                    sample_idx = np.random.choice(regime_indices, min(50, len(regime_indices)), replace=False)
                    val_df = pd.concat([val_df, df_copy.loc[sample_idx]])

                if regime not in test_regimes and len(regime_indices) > 0:
                    sample_idx = np.random.choice(regime_indices, min(50, len(regime_indices)), replace=False)
                    test_df = pd.concat([test_df, df_copy.loc[sample_idx]])

        # Remove auxiliary columns used for regime detection
        for split_df in [train_df, val_df, test_df]:
            if 'returns' in split_df.columns:
                split_df.drop(columns=['returns'], inplace=True)
            if 'volatility' in split_df.columns:
                split_df.drop(columns=['volatility'], inplace=True)

        return {
            'train': train_df.sort_index(),
            'val': val_df.sort_index(),
            'test': test_df.sort_index()
        }, {
            'regime_counts': regime_counts,
            'regimes_in_train': list(train_df['regime'].unique()),
            'regimes_in_val': list(val_df['regime'].unique()),
            'regimes_in_test': list(test_df['regime'].unique())
        }

    def prepare_model_inputs(self, splits: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Prepare model inputs with sequences and targets
        
        Args:
            splits: Dictionary with split DataFrames
            
        Returns:
            Dictionary with X and y data for each split
        """
        window_size = self.config.get('window_size', 60)
        feature_cols = [col for col in splits['train'].columns 
                        if col not in ['regime']]
        
        result = {}
        
        for split_name, split_df in splits.items():
            X_sequences = []
            y_price = []
            
            # Remove regime column for input features if present
            X_data = split_df[feature_cols]
            
            # Create sequences
            for i in range(len(X_data) - window_size):
                X_sequences.append(X_data.iloc[i:i+window_size].values)
                # Target is the next close price
                y_price.append(X_data.iloc[i+window_size]['Close'])
            
            if X_sequences:
                result[split_name] = {
                    'X': np.array(X_sequences),
                    'y': np.array(y_price)
                }
            else:
                logger.warning(f"No sequences could be created for {split_name} split")
                result[split_name] = {
                    'X': np.array([]),
                    'y': np.array([])
                }
                
        return result

    def process_data(self, csv_path: str) -> Dict:
        """
        Main method to process data end-to-end

        Args:
            csv_path: Path to CSV file

        Returns:
            Dictionary with processed data
        """
        try:
            # Load data
            df = self.load_data(csv_path)
            if df is None:
                raise ValueError("Failed to load data")

            # Check if dataframe is empty
            if df.empty or len(df) < 10:  # Ensure we have at least some minimum number of rows
                raise ValueError("Dataset is empty or too small for processing")

            # Preprocess
            df = self.preprocess(df)
            logger.info(f"After preprocessing: DataFrame shape = {df.shape}")

            # Normalize
            df_normalized = self.normalize_features(df)
            logger.info(f"After normalization: DataFrame shape = {df_normalized.shape}")

            # Resample if needed
            resampled_data = self.resample_data(df_normalized)

            # Create splits for each timeframe
            result = {}
            for timeframe, timeframe_df in resampled_data.items():
                logger.info(f"Processing {timeframe} data with shape {timeframe_df.shape}")

                # Create regime-aware splits
                splits, regime_info = self.create_regime_aware_split(timeframe_df)

                # Prepare inputs for model
                model_inputs = self.prepare_model_inputs(splits)

                result[timeframe] = {
                    'splits': splits,
                    'model_inputs': model_inputs,
                    'regime_info': regime_info,
                    'feature_scaler': self.scaler
                }

            return result
        except Exception as e:
            logger.error(f"Error in process_data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def generate_mock_data(self, sample_row, num_rows=1000):
        """Generate mock data based on a sample row"""
        # Parse sample row
        parts = sample_row.split(',')
        date, time, open_price, high, low, close, volume, ema9, ema21, ema220 = parts

        # Create base dataframe
        base_date = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M:%S")
        dates = [base_date + timedelta(minutes=i) for i in range(num_rows)]

        # Generate random price movements
        np.random.seed(42)  # For reproducibility
        close_prices = [float(close)]
        for i in range(1, num_rows):
            # Random walk with drift
            change = np.random.normal(0.01, 0.1)  # Mean positive drift
            new_price = close_prices[-1] * (1 + change)
            close_prices.append(new_price)

        # Generate OHLC data
        df = pd.DataFrame({
            'Date': [d.strftime("%Y-%m-%d") for d in dates],
            'Time': [d.strftime("%H:%M:%S") for d in dates],
            'Open': [p * (1 + np.random.normal(0, 0.001)) for p in close_prices],
            'High': [p * (1 + abs(np.random.normal(0, 0.002))) for p in close_prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.002))) for p in close_prices],
            'Close': close_prices,
            'Volume': [int(abs(np.random.normal(int(volume), int(volume) * 0.1))) for _ in range(num_rows)],
        })

        # Ensure High is the highest price
        df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
        # Ensure Low is the lowest price
        df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)

        # Calculate EMAs
        df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
        df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
        df['EMA220'] = df['Close'].ewm(span=220, adjust=False).mean()

        return df
