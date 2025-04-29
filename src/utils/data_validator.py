from typing import Dict, Any, List
import numpy as np
import pandas as pd

class DataValidator:
    def __init__(self, required_features: List[str], feature_ranges: Dict[str, tuple]):
        self.required_features = required_features
        self.feature_ranges = feature_ranges

    def validate_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input data against requirements"""
        # Check required features
        missing_features = [f for f in self.required_features if f not in data]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        # Validate ranges
        for feature, value in data.items():
            if feature in self.feature_ranges:
                min_val, max_val = self.feature_ranges[feature]
                if not min_val <= float(value) <= max_val:
                    raise ValueError(
                        f"Feature {feature} value {value} outside valid range [{min_val}, {max_val}]"
                    )

        return data