from typing import Dict, Any, Optional
import joblib
from pathlib import Path
import hashlib

class ModelCache:
    _instance = None
    _models: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelCache, cls).__new__(cls)
        return cls._instance

    def get_model(self, model_path: str) -> Any:
        """Get model from cache or load it"""
        model_hash = self._get_file_hash(model_path)
        
        if model_hash not in self._models:
            self._models[model_hash] = joblib.load(model_path)
        
        return self._models[model_hash]

    @staticmethod
    def _get_file_hash(file_path: str) -> str:
        """Calculate file hash for cache key"""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def clear_cache(self):
        """Clear the model cache"""
        self._models.clear()