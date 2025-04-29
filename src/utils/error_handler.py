from typing import Dict, Any, Optional
import logging

class MedExplainError(Exception):
    """Base exception class for MedExplain"""
    pass

class ModelError(MedExplainError):
    """Exception raised for model-related errors"""
    pass

class DataValidationError(MedExplainError):
    """Exception raised for data validation errors"""
    pass

def handle_prediction_error(func):
    """Decorator for handling prediction errors"""
    def wrapper(*args, **kwargs) -> Dict[str, Any]:
        try:
            return func(*args, **kwargs)
        except ValueError as ve:
            logging.error(f"Validation error: {str(ve)}")
            return {
                "error": f"Invalid input data: {str(ve)}",
                "prediction": None,
                "probability": None
            }
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            return {
                "error": f"Prediction error: {str(e)}",
                "prediction": None,
                "probability": None
            }
    return wrapper