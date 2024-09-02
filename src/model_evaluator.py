import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn.metrics import balanced_accuracy_score

from sklearn.base import BaseEstimator


class ModelEvaluator(BaseEstimator):
    def __init__(self, model_trainer, metrics=None, f1_average='macro'):
      
        self.model_trainer = model_trainer
        self.metrics = metrics or []
        self.f1_average = f1_average

    def fit(self, X_train, y_train):

        self.model_trainer.fit(X_train, y_train)
        return self

    def evaluate(self, X_test, y_test):

        y_pred = self.model_trainer.predict(X_test)

        metrics_results = {}

        if 'accuracy' in self.metrics:
            metrics_results['accuracy'] = accuracy_score(y_test, y_pred)

        if 'precision' in self.metrics:
            metrics_results['precision'] = precision_score(y_test, y_pred, average=self.f1_average)

        if 'recall' in self.metrics:
            metrics_results['recall'] = recall_score(y_test, y_pred, average=self.f1_average)

        if 'f1_macro' in self.metrics or 'f1_micro' in self.metrics or 'f1_weighted' in self.metrics:
            metrics_results['f1'] = f1_score(y_test, y_pred, average=self.f1_average)

        return metrics_results

    def evaluate_to_dataframe(self, X_test, y_test):

        results = self.evaluate(X_test, y_test)
        df_results = pd.DataFrame(list(results.items()), columns=['Metric', 'Score'])
        return df_results