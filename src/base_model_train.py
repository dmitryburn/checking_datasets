import pandas as pd
import lightgbm as lgb
from sklearn.base import BaseEstimator, TransformerMixin

class BaseModelTrain(BaseEstimator, TransformerMixin):
    def __init__(self, model, columns_to_delete=None, categorical_features=None, **model_params):
        """
        Инициализирует класс с моделью, колонками для удаления, категориальными признаками и гиперпараметрами модели.
        
        :param model: Модель для обучения. Ожидается, что это объект LightGBM.
        :param columns_to_delete: Список колонок для удаления перед обучением. Если None, то колонки не удаляются.
        :param categorical_features: Список колонок, которые являются категориальными.
        :param model_params: Гиперпараметры модели.
        """
        self.columns_to_delete = columns_to_delete or []
        self.categorical_features = categorical_features or []
        self.model = model.set_params(**model_params) if model_params else model

    def fit(self, X, y):
        X_transformed = X.drop(columns=self.columns_to_delete, errors='ignore')
        
        for cat_feature in self.categorical_features:
            if cat_feature in X_transformed.columns:
                X_transformed[cat_feature] = X_transformed[cat_feature].astype('category')
        
        

        self.model.fit(X_transformed, y, categorical_feature=self.categorical_features)
        return self

    def predict(self, X):
        X_transformed = X.drop(columns=self.columns_to_delete, errors='ignore')
        
        for cat_feature in self.categorical_features:
            if cat_feature in X_transformed.columns:
                X_transformed[cat_feature] = X_transformed[cat_feature].astype('category')
        
        return self.model.predict(X_transformed)
 
    def set_params(self, **params):

        model_params = {key: value for key, value in params.items() if key != 'model' and key != 'categorical_features'}
        self.model.set_params(**model_params)
        
        if 'categorical_features' in params:
            self.categorical_features = params['categorical_features']
        
        return self