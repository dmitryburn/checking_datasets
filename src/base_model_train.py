import pandas as pd
import lightgbm as lgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputClassifier

class BaseModelTrain(BaseEstimator, TransformerMixin):
    def __init__(self, model, columns_to_delete=None, categorical_features=None, **model_params):
        """
        Инициализирует класс с моделью, колонками для удаления, категориальными признаками и гиперпараметрами модели.
        
        :param model: Модель для обучения. Может быть как обычным классификатором, так и MultiOutputClassifier для мультитаргет задачи.
        :param columns_to_delete: Список колонок для удаления перед обучением. Если None, то колонки не удаляются.
        :param categorical_features: Список колонок, которые являются категориальными.
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
    

    def fit(self, X, y):

        
        X_transformed = X.drop(columns=self.columns_to_delete, errors='ignore')
        
        # Преобразуем категориальные признаки в категорию заранее
        for cat_feature in self.categorical_features:
            if cat_feature in X_transformed.columns:
                X_transformed[cat_feature] = X_transformed[cat_feature].astype('category')
        
        # Для MultiOutputClassifier убираем передачу categorical_feature
        if self.task_type == 'multilabel':
            self.model.fit(X_transformed, y)  # categorical_feature не используется
        else:
            self.model.fit(X_transformed, y, categorical_feature=self.categorical_features)
        
        return self

    def predict(self, X):

        
        X_transformed = X.drop(columns=self.columns_to_delete, errors='ignore')
        
        for cat_feature in self.categorical_features:
            if cat_feature in X_transformed.columns:
                X_transformed[cat_feature] = X_transformed[cat_feature].astype('category')
        
        return self.model.predict(X_transformed)

    def set_params(self, **params):
        model_params = {key: value for key, value in params.items() if key not in ['model', 'categorical_features']}
        
        if self.task_type == 'multilabel':
            self.model.estimator.set_params(**model_params)
        else:
            self.model.set_params(**model_params)
        
        if 'categorical_features' in params:
            self.categorical_features = params['categorical_features']
        
        return self
