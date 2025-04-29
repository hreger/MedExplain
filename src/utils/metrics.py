from typing import Dict, Any, List
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
from .config import Config

class MetricsTracker:
    def __init__(self):
        self.config = Config()
        self._setup_mlflow()

    def _setup_mlflow(self):
        """Configure MLflow tracking"""
        mlflow.set_tracking_uri(self.config.get('mlflow.tracking_uri', 'mlruns'))
        mlflow.set_experiment(self.config.get('mlflow.experiment_name', 'MedExplain'))

    def log_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                   additional_metrics: Dict[str, Any] = None) -> Dict[str, float]:
        """Log prediction metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }

        if additional_metrics:
            metrics.update(additional_metrics)

        # Log to MLflow
        with mlflow.start_run(nested=True):
            mlflow.log_metrics(metrics)

        return metrics