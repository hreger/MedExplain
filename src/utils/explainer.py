from typing import Dict, Any, List
import numpy as np
import shap
import lime
import lime.lime_tabular
from .config import Config
from .error_handler import ModelError

class ExplainabilityManager:
    def __init__(self, model, feature_names: List[str]):
        self.config = Config()
        self.model = model
        self.feature_names = feature_names
        self._setup_explainers()

    def _setup_explainers(self):
        """Initialize SHAP and LIME explainers"""
        try:
            self.shap_explainer = shap.TreeExplainer(self.model)
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=np.array([]),  # Will be set during explanation
                feature_names=self.feature_names,
                class_names=['Negative', 'Positive'],
                mode='classification'
            )
        except Exception as e:
            raise ModelError(f"Failed to initialize explainers: {str(e)}")

    def explain_prediction(self, data: np.ndarray) -> Dict[str, Any]:
        """Generate SHAP and LIME explanations"""
        try:
            # Generate SHAP values
            shap_values = self.shap_explainer.shap_values(data)
            
            # Generate LIME explanation
            lime_exp = self.lime_explainer.explain_instance(
                data[0], 
                self.model.predict_proba,
                num_features=len(self.feature_names)
            )

            return {
                'shap_values': shap_values,
                'feature_importance': dict(zip(self.feature_names, np.abs(shap_values).mean(0))),
                'lime_explanation': lime_exp.as_list()
            }
        except Exception as e:
            raise ModelError(f"Failed to generate explanations: {str(e)}")