import yaml
from pathlib import Path
from typing import Dict, Any

class Config:
    _instance = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        """Load configuration from YAML file"""
        config_path = Path(__file__).parent.parent.parent / 'config.yaml'
        if config_path.exists():
            with open(config_path) as f:
                self._config = yaml.safe_load(f)
        else:
            self._config = self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'model': {
                'threshold': 0.5,
                'cache_enabled': True
            },
            'features': {
                'ranges': {
                    'glucose': (0, 500),
                    'bmi': (10, 100),
                    'age': (0, 120)
                }
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self._config.get(key, default)