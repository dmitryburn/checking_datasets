import pandas as pd
from sklearn.metrics import (f1_score, precision_score, recall_score, 
                             accuracy_score, hamming_loss, roc_auc_score, 
                             jaccard_score)
from sklearn.base import BaseEstimator
import numpy as np
from typing import List, Optional, Dict, Union

class ModelEvaluator(BaseEstimator):
    """
    Класс для оценки модели на основе выбранных метрик.

    Параметры:
    - model_trainer: Объект модели, который реализует метод fit и predict.
    - metrics: Список метрик для оценки модели. Поддерживаемые метрики: 'accuracy', 'precision', 'recall', 'f1_macro', 'f1_micro', 'f1_weighted', 'f1_samples', 'hamming_loss', 'jaccard_score', 'roc_auc'.
    - f1_average: Параметр для усреднения F1-метрики. Может быть 'macro', 'micro', 'weighted'.
    - base_multitarget: Флаг, указывающий, нужно ли учитывать многоцелевое прогнозирование.
    """
    
    def __init__(self, model_trainer, metrics: Optional[List[str]] = None, f1_average: str = 'macro', base_multitarget: bool = False):
        """
        Инициализирует объект ModelEvaluator.

        :param model_trainer: Объект модели, реализующий методы fit и predict.
        :param metrics: Список метрик для оценки.
        :param f1_average: Метод усреднения для F1-метрики ('macro', 'micro', 'weighted').
        :param base_multitarget: Флаг для поддержки многоцелевого прогнозирования.
        """
        self.model_trainer = model_trainer
        self.metrics = metrics or []
        self.f1_average = f1_average
        self.base_multitarget = base_multitarget

    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> 'ModelEvaluator':
        """
        Обучает модель на предоставленных данных.

        :param X_train: Признаки для обучения.
        :param y_train: Целевые переменные для обучения.
        :return: Self
        """
        self.model_trainer.fit(X_train, y_train)
        return self

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict[str, float]:
        """
        Оценивает модель на тестовом наборе данных по указанным метрикам.

        :param X_test: Признаки для тестирования.
        :param y_test: Целевые переменные для тестирования.
        :return: Словарь с результатами метрик.
        """
        is_multioutput = len(y_test.shape) > 1 and y_test.shape[1] > 1
        
        if is_multioutput:
            multitarget_col_names = y_test.columns
            y_pred = self.model_trainer.predict(X_test).astype(int)
            y_test = np.array(y_test).astype(int)
        else:
            y_pred = self.model_trainer.predict(X_test)

        metrics_results = {}

        if is_multioutput and self.base_multitarget:
            metrics_results = pd.DataFrame(index=range(y_test.shape[1]), columns=self.metrics)
            metrics_results['Targets']  = multitarget_col_names
            for i in range(y_test.shape[1]):
                y_test_i = y_test[:, i]
                y_pred_i = y_pred[:, i]

                row_metrics = {}
                if 'accuracy' in self.metrics:
                    row_metrics['accuracy'] = accuracy_score(y_test_i, y_pred_i)
                if 'precision' in self.metrics:
                    row_metrics['precision'] = precision_score(y_test_i, y_pred_i, average=self.f1_average, zero_division=0)
                if 'recall' in self.metrics:
                    row_metrics['recall'] = recall_score(y_test_i, y_pred_i, average=self.f1_average, zero_division=0)
                if 'f1_macro' in self.metrics:
                    row_metrics['f1_macro'] = f1_score(y_test_i, y_pred_i, average='macro')

                for metric in row_metrics:
                    metrics_results.at[i, metric] = row_metrics[metric]
        else:
            if 'accuracy' in self.metrics:
                if is_multioutput:
                    metrics_results['accuracy'] = 1 - hamming_loss(y_test, y_pred)
                else:
                    metrics_results['accuracy'] = accuracy_score(y_test, y_pred)

            if 'precision' in self.metrics:
                metrics_results['precision'] = precision_score(y_test, y_pred, average=self.f1_average, zero_division=0)

            if 'recall' in self.metrics:
                metrics_results['recall'] = recall_score(y_test, y_pred, average=self.f1_average, zero_division=0)

            if 'f1_macro' in self.metrics or 'f1_micro' in self.metrics or 'f1_weighted' in self.metrics:
                metrics_results['f1'] = f1_score(y_test, y_pred, average=self.f1_average)

            if 'f1_samples' in self.metrics:
                metrics_results['f1_samples'] = f1_score(y_test, y_pred, average='samples')

            if 'hamming_loss' in self.metrics:
                metrics_results['hamming_loss'] = hamming_loss(y_test, y_pred)

            if 'jaccard_score' in self.metrics:
                metrics_results['jaccard_score'] = jaccard_score(y_test, y_pred, average='samples')

            if 'roc_auc' in self.metrics:
                if is_multioutput:
                    roc_auc_scores = []
                    for i in range(y_test.shape[1]):
                        roc_auc_scores.append(
                            roc_auc_score(y_test[:, i], y_pred[:, i])
                        )
                    metrics_results['roc_auc'] = np.mean(roc_auc_scores)
                else:
                    metrics_results['roc_auc'] = roc_auc_score(y_test, y_pred, average=self.f1_average, multi_class='ovr')

        return metrics_results

    def evaluate_to_dataframe(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> pd.DataFrame:
        """
        Оценивает модель и возвращает результаты в формате pandas DataFrame.

        :param X_test: Признаки для тестирования.
        :param y_test: Целевые переменные для тестирования.
        :return: pandas DataFrame с результатами оценки.
        """
        results = self.evaluate(X_test, y_test)
        if self.base_multitarget and isinstance(results, pd.DataFrame):
            return results
        else:
            df_results = pd.DataFrame(list(results.items()), columns=['Metric', 'Score'])
            return df_results     