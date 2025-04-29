from pathlib import Path
import joblib
import hashlib
import logging
from typing import Any, Optional
from .error_handler import ModelError
from .config import Config

class ModelManager:
    def __init__(self):
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        self._model_cache = {}

    def load_model(self, model_path: str) -> Any:
        """Load model with integrity check and caching"""
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                raise ModelError(f"Model file not found: {model_path}")

            # Check model integrity
            model_hash = self._calculate_hash(model_path)
            
            # Return cached model if available
            if self.config.get('model.cache_enabled', True):
                if model_hash in self._model_cache:
                    return self._model_cache[model_hash]

            # Load and cache model
            model = joblib.load(model_path)
            self._model_cache[model_hash] = model
            return model

        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise ModelError(f"Failed to load model: {str(e)}")

    def _calculate_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of model file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()