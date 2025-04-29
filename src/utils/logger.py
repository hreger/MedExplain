import logging
from pathlib import Path
from typing import Optional
from .config import Config

def setup_logging(log_file: Optional[str] = None):
    """Setup logging configuration"""
    config = Config()
    log_config = config.get('logging', {})
    
    # Create logs directory if it doesn't exist
    if log_file:
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / log_file
    
    logging.basicConfig(
        level=log_config.get('level', 'INFO'),
        format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        filename=log_file
    )