import pandas as pd
import os
import yaml
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Union, Optional
from enum import Enum
from dotenv import load_dotenv
from oianalytics.api import (
                get_multiple_data_values,
                set_default_oianalytics_credentials,
                OIAnalyticsAPICredentials
            )

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSourceType(Enum):
    """Enum for the supported data source types."""
    OIANALYTICS = "oianalytics"
    CSV = "csv"


class TimeSeriesDataLoader:
    """
    A class for loading time series data from different sources.
    
    This class supports loading data from:
    - OIAnalytics API
    - CSV files
    
    Configuration is from YAML files, and sensitive credentials from .env.
    """
    
    def __init__(self,
                 config_path: str,
                 source_type: DataSourceType = DataSourceType.CSV):
        """
        Initialize the TimeSeriesDataLoader.
        
        Args:
            config_path: Path to the YAML configuration file
            source_type: Type of data source to use (default: CSV)
        """
        self.config_path = config_path
        self.source_type = source_type
        self.tags = []
        self.duration_days = 1  # Default duration
        self.config = self.load_config()
        
    def load_config(self) -> Dict:
        """
        Load and process the YAML configuration file.
        
        Returns:
            Dictionary containing configuration options
        """
        logger.info(f"Loading configuration from {self.config_path}")
        
        try:
            # Load YAML configuration
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            # Extract tags and settings
            self.tags = config.get('tags', [])
            self.duration_days = config.get('duration_days', 365)
            
            # Extract date settings
            self.start_date = config.get('start_date')
            self.end_date = config.get('end_date')
            
            logger.info(f"Loaded {len(self.tags)} tags from configuration")
            # logger.info(f"Duration set to {self.duration_days} days")
            
            return config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise
            
    def load_data(self,
                  start_time: Optional[Union[str, datetime]] = None,
                  end_time: Optional[Union[str, datetime]] = None
                  ) -> pd.DataFrame:
        """
        Load time series data based on the configured source type.
        
        Args:
            start_time: Start time for the data (default: end_time - duration)
            end_time: End time for the data (default: current time)
        Returns:
            DataFrame containing the time series data
        """
        # Use config dates if no dates provided
        if not end_time:
            end_time = self.end_date if self.end_date else datetime.now()
        if not start_time:
            start_time = (
                self.start_date if self.start_date 
                else (self._parse_time(end_time) - timedelta(days=self.duration_days))
            )
        
        logger.info(f"Loading data from {start_time} to {end_time}")
        
        logger.info(f"Loading {len(self.tags)} tags")
            
        # Load data based on source type
        if self.source_type == DataSourceType.OIANALYTICS:
            return self._load_from_oianalytics(
                self.tags, start_time, end_time)
        elif self.source_type == DataSourceType.CSV:
            return self._load_from_csv(self.tags)
        else:
            msg = f"Unsupported data source type: {self.source_type}"
            raise ValueError(msg)
            
    def _parse_time(self, time_val: Union[str, datetime]) -> datetime:
        """Parse time value to datetime."""
        if isinstance(time_val, datetime):
            return time_val
        else:
            return pd.to_datetime(time_val)
            
    def _load_from_oianalytics(self,
                               tags: List[str],
                               start_time: datetime,
                               end_time: datetime) -> pd.DataFrame:
        """Load data from OIAnalytics API."""
        try:
            # Get credentials from environment variables
            base_url = os.getenv("OIANALYTICS_BASE_URL")
            login = os.getenv("OIANALYTICS_LOGIN")
            pwd = os.getenv("OIANALYTICS_PWD")
            
            # Set up credentials
            oia_creds = OIAnalyticsAPICredentials(base_url, login, pwd)
            set_default_oianalytics_credentials(oia_creds)
            
            # Set aggregation settings from config
            aggr_period = pd.Timedelta(minutes=5).isoformat()
            aggr_function = self.config.get('aggregation_function', 'MEAN')
            aggregation = self.config.get('aggregation', 'TIME')
            
            # Request data from API
            time_df = get_multiple_data_values(
                data_reference=tags,
                start_date=start_time,
                end_date=end_time,
                aggregation=aggregation,
                aggregation_period=aggr_period,
                aggregation_function=aggr_function
            )
            
            logger.info(f"Loaded data from OIAnalytics: shape {time_df.shape}")
            return time_df
            
        except Exception as e:
            msg = f"Error loading data from OIAnalytics: {str(e)}"
            logger.error(msg)
            raise
            
    def _load_from_csv(self, tags: List[str]) -> pd.DataFrame:
        """Load data from CSV files."""
        try:
            # Get CSV path from config or environment variable
            data_path = self.config.get('csv_path')
            if not data_path:
                raise ValueError("csv_path must be provided in config")
                
            # Read the CSV file
            df = pd.read_csv(data_path)
            
            # Ensure timestamp column is datetime
            time_col = self.config.get('time_column', 'timestamp')
            if time_col in df.columns:
                df[time_col] = pd.to_datetime(df[time_col])
                df.set_index(time_col, inplace=True)
            
            # Filter columns to include only the specified tags
            tag_cols = [col for col in df.columns if col in tags]
            if not tag_cols:
                msg = "None of the specified tags found in CSV columns"
                logger.warning(msg)
                
            filtered_df = df[tag_cols]
            logger.info(f"Loaded data from CSV: shape {filtered_df.shape}")
            return filtered_df
            
        except Exception as e:
            logger.error(f"Error loading data from CSV: {str(e)}")
            raise