import numpy as np
from loguru import logger

try:
    from lime.lime_tabular import LimeTabularExplainer
except ImportError:
    logger.warning("LIME not installed. Run `pip install lime`.")
    LimeTabularExplainer = None

class LIMEExplainer:
    """
    Paper Sec. 3.6: Local Interpretable Model-agnostic Explanations.
    Explains individual DoH anomaly predictions.
    """
    def __init__(self, predict_fn, feature_names: list[str], num_samples: int = 5000):
        self.predict_fn = predict_fn
        self.feature_names = feature_names
        self.num_samples = num_samples
        self._explainer = None

    def fit(self, background_data: np.ndarray):
        if LimeTabularExplainer is None:
            raise RuntimeError("lime package is required.")
            
        self._explainer = LimeTabularExplainer(
            training_data=background_data,
            feature_names=self.feature_names,
            mode="classification",
            discretize_continuous=True,
            random_state=42
        )
        logger.info("LIME Tabular Explainer fitted.")

    def explain(self, x: np.ndarray, num_features: int = 10) -> dict:
        if self._explainer is None:
            raise RuntimeError("Call fit() before explain().")
            
        exp = self._explainer.explain_instance(
            data_row=x.reshape(-1),
            predict_fn=lambda v: np.stack(
                [1 - self.predict_fn(v), self.predict_fn(v)], axis=-1
            ),
            num_samples=self.num_samples,
            num_features=num_features
        )
        
        return {
            "feature_weights": exp.as_list(),
            "intercept": exp.intercept[1],
            "prediction_local": exp.local_pred,
            "method": "LIME"
        }
