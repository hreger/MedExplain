import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union
from .data_validator import DataValidator
from .config import Config

class DataPreprocessor:
    def __init__(self):
        self.config = Config()
        self.validator = DataValidator(
            required_features=self.config.get('features.required', []),
            feature_ranges=self.config.get('features.ranges', {})
        )

    def preprocess(self, data: Union[Dict[str, Any], pd.DataFrame]) -> pd.DataFrame:
        """Preprocess input data"""
        # Convert dict to DataFrame if necessary
        if isinstance(data, dict):
            data = pd.DataFrame([data])

        # Validate data
        self._validate_data(data)

        # Handle missing values
        data = self._handle_missing_values(data)

        # Feature scaling
        data = self._scale_features(data)

        return data

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data"""
        for _, row in data.iterrows():
            self.validator.validate_input(row.to_dict())

    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        # Get strategy from config
        strategy = self.config.get('preprocessing.missing_values', 'mean')
        
        if strategy == 'mean':
            return data.fillna(data.mean())
        elif strategy == 'median':
            return data.fillna(data.median())
        elif strategy == 'zero':
            return data.fillna(0)
        else:
            return data.dropna()

    def _scale_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Scale features based on configuration"""
        scaling_config = self.config.get('preprocessing.scaling', None)
        if not scaling_config:
            return data

        # Apply scaling based on config
        if scaling_config == 'standard':
            return (data - data.mean()) / data.std()
        elif scaling_config == 'minmax':
            return (data - data.min()) / (data.max() - data.min())
        
        return data