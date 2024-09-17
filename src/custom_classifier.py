import numpy as np
import lightgbm as lgb
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import List

class MultiTargetLGBMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_targets: int, is_unbalance: bool = True, boosting_type: str = 'gbdt', 
                 objective: str = 'binary', metric: str = 'binary_error', **kwargs):
        """
        Инициализация класса MultiTargetLGBMClassifier.

        Parameters:
        - n_targets: Количество целевых переменных (для создания соответствующего числа моделей).
        - is_unbalance: Использовать ли автоматическую балансировку классов.
        - boosting_type: Тип бустинга для модели LightGBM.
        - objective: Целевая функция для обучения модели LightGBM.
        - metric: Метрика для оценки модели LightGBM.
        - **kwargs: Дополнительные параметры для LightGBM.
        """
        self.n_targets = n_targets
        self.is_unbalance = is_unbalance
        self.boosting_type = boosting_type
        self.objective = objective
        self.metric = metric
        self.kwargs = kwargs
        self.models = [lgb.LGBMClassifier(verbose=-1, is_unbalance=self.is_unbalance, 
                                          boosting_type=self.boosting_type, 
                                          objective=self.objective, 
                                          metric=self.metric, 
                                          **self.kwargs) for _ in range(self.n_targets)]
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MultiTargetLGBMClassifier':
        """
        Обучение модели для каждой целевой переменной.

        Parameters:
        - X: Входные данные для обучения.
        - y: Целевые переменные для обучения.
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y must be the same.")
        
        if len(y.shape) != 2 or y.shape[1] != self.n_targets:
            raise ValueError(f"y should be a 2D array with {self.n_targets} columns.")
        
        for ind, model in enumerate(self.models):
            model.fit(X, y[:, ind])
        
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Прогнозирование для каждой целевой переменной.

        Parameters:
        - X: Входные данные для прогнозирования.
        
        Returns:
        - Предсказанные значения для всех целевых переменных.
        """
        predictions = []
        for model in self.models:
            predictions.append(model.predict(X))
        
        return np.array(predictions).T

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Получение вероятностей прогнозирования для каждой целевой переменной.

        Parameters:
        - X: Входные данные для прогнозирования.
        
        Returns:
        - Вероятности для всех целевых переменных.
        """
        prob_predictions = []
        for model in self.models:
            prob_predictions.append(model.predict_proba(X)[:, 1]) 
        
        return np.array(prob_predictions).T