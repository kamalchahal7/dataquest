"""
sleep_wrapper.py  —  SleepClassifierWrapper

Defined in its own module so that pickle can resolve
`sleep_wrapper.SleepClassifierWrapper` regardless of which
script created the model.pkl.
"""

import numpy as np


def _quality_score_to_label(score: float) -> str:
    if score <= 6:
        return "Poor"
    elif score <= 7.5:
        return "Moderate"
    return "Good"


class SleepClassifierWrapper:
    """Wraps the GradientBoosting sleep quality regressor to expose a classifier interface.

    predict_proba distributes probability via a Gaussian centred on each
    quality-band midpoint, so the highest-probability class corresponds to
    whichever band the regression score falls closest to.
    """

    CLASSES = ["Poor", "Moderate", "Good"]
    _CENTERS = [4.0, 6.75, 8.5]
    _SHARPNESS = 1.8

    def __init__(self, regressor_pipeline):
        self.regressor = regressor_pipeline
        self.classes_ = self.CLASSES

    @classmethod
    def score_to_label(cls, score: float) -> str:
        return _quality_score_to_label(float(score))

    def predict(self, X):
        scores = self.regressor.predict(X)
        return np.array([self.score_to_label(s) for s in scores])

    def predict_proba(self, X):
        scores = self.regressor.predict(X)
        proba_rows = []
        for score in scores:
            score = float(np.clip(score, 1.0, 10.0))
            weights = [np.exp(-self._SHARPNESS * abs(score - c)) for c in self._CENTERS]
            total = sum(weights)
            proba_rows.append([w / total for w in weights])
        return np.array(proba_rows)
