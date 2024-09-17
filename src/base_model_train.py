import pandas as pd
import lightgbm as lgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputClassifier

class BaseModelTrain(BaseEstimator, TransformerMixin):
    """
    Базовый класс для обучения моделей с возможностью удаления признаков и обработки категориальных признаков.

    Этот класс можно использовать как для задач с одним целевым признаком, так и для задач с несколькими целевыми признаками.
    Он позволяет указать столбцы, которые следует удалить перед обучением, и явно обработать категориальные признаки.

    Параметры:
    - model: Модель для обучения. Это может быть обычный классификатор или MultiOutputClassifier для многозадачных задач.
    - columns_to_delete: Список столбцов, которые будут удалены из набора данных перед обучением. По умолчанию None.
    - categorical_features: Список имен столбцов, которые являются категориальными признаками. По умолчанию None.
    - **model_params: Гиперпараметры модели.
    """
    
    def __init__(self, model, columns_to_delete=None, categorical_features=None, **model_params):
        """
        Инициализирует класс BaseModelTrain.

        :param model: Модель для обучения (например, lightgbm.LGBMClassifier или sklearn.multioutput.MultiOutputClassifier).
        :param columns_to_delete: Список столбцов, которые будут удалены из набора данных перед обучением. Если None, то удаление не выполняется.
        :param categorical_features: Список имен столбцов, которые являются категориальными признаками. Если None, то обработка категориальных признаков не выполняется.
        :param model_params: Гиперпараметры модели.
        """
        self.columns_to_delete = columns_to_delete or []
        self.categorical_features = categorical_features or []
        self.model = model

        if isinstance(self.model, MultiOutputClassifier):
            self.task_type = 'multilabel'
        else:
            self.task_type = 'singlelabel'

        self.set_params(**model_params)
    

    def fit(self, X: pd.DataFrame, y) -> 'BaseModelTrain':
        """
        Обучает модель на данных.

        :param X: Матрица признаков (pandas DataFrame).
        :param y: Целевая переменная. Может быть одиночной целевой переменной для задач с одним признаком или несколькими целевыми переменными для многозадачных задач.
        :return: self
        """

        X_transformed = X.drop(columns=self.columns_to_delete, errors='ignore')
        
        for cat_feature in self.categorical_features:
            if cat_feature in X_transformed.columns:
                X_transformed[cat_feature] = X_transformed[cat_feature].astype('category')
        
        if self.task_type == 'multilabel':
            self.model.fit(X_transformed, y)
        else:
            self.model.fit(X_transformed, y, categorical_feature=self.categorical_features)
        
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Выполняет предсказание с помощью обученной модели.

        :param X: Матрица признаков (pandas DataFrame).
        :return: Прогнозы в виде pandas Series или DataFrame, в зависимости от типа задачи.
        """

        X_transformed = X.drop(columns=self.columns_to_delete, errors='ignore')
        
        for cat_feature in self.categorical_features:
            if cat_feature in X_transformed.columns:
                X_transformed[cat_feature] = X_transformed[cat_feature].astype('category')
        
        return self.model.predict(X_transformed)

    def set_params(self, **params) -> 'BaseModelTrain':
        """
        Устанавливает параметры для модели.

        :param params: Гиперпараметры модели.
        :return: self
        """
        model_params = {key: value for key, value in params.items() if key not in ['model', 'categorical_features']}

        if self.task_type == 'multilabel':
            self.model.estimator.set_params(**model_params)
        else:
            self.model.set_params(**model_params)
        
        if 'categorical_features' in params:
            self.categorical_features = params['categorical_features']
        
        return self
