import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn.metrics import balanced_accuracy_score

from sklearn.metrics import (f1_score, precision_score, recall_score, 
                             hamming_loss, roc_auc_score)
from sklearn.base import BaseEstimator
import numpy as np
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
        y_test = np.array(y_test)

        metrics_results = {}
        
        # Check if the problem is multi-output
        is_multioutput = len(y_test.shape) > 1 and y_test.shape[1] > 1
        print(is_multioutput)

        if 'accuracy' in self.metrics:
            if is_multioutput:
                metrics_results['accuracy'] = 1 - hamming_loss(y_test, y_pred)
            else:
                metrics_results['accuracy'] = accuracy_score(y_test, y_pred)

        if 'precision' in self.metrics:
            if is_multioutput:
                # Compute precision for each label separately
                precision_scores = []
                for i in range(y_test.shape[1]):
                    precision_scores.append(
                        precision_score(y_test[:, i], y_pred[:, i], average=self.f1_average, zero_division=0)
                    )
                metrics_results['precision'] = np.mean(precision_scores)
            else:
                metrics_results['precision'] = precision_score(y_test, y_pred, average=self.f1_average, zero_division=0)

        if 'recall' in self.metrics:
            if is_multioutput:
                # Compute recall for each label separately
                recall_scores = []
                for i in range(y_test.shape[1]):
                    recall_scores.append(
                        recall_score(y_test[:, i], y_pred[:, i], average=self.f1_average, zero_division=0)
                    )
                metrics_results['recall'] = np.mean(recall_scores)
            else:
                metrics_results['recall'] = recall_score(y_test, y_pred, average=self.f1_average, zero_division=0)

        if 'f1_macro' in self.metrics or 'f1_micro' in self.metrics or 'f1_weighted' in self.metrics:
            if is_multioutput:
                # Compute F1 score for each label separately
                f1_scores = []
                for i in range(y_test.shape[1]):
                    f1_scores.append(
                        f1_score(y_test[:, i], y_pred[:, i], average=self.f1_average)
                    )
                metrics_results['f1'] = np.mean(f1_scores)
            else:
                metrics_results['f1'] = f1_score(y_test, y_pred, average=self.f1_average)

        if 'hamming_loss' in self.metrics:
            metrics_results['hamming_loss'] = hamming_loss(y_test, y_pred)

        if 'roc_auc' in self.metrics:
            if is_multioutput:
                # Compute ROC AUC for each label separately
                roc_auc_scores = []
                for i in range(y_test.shape[1]):
                    roc_auc_scores.append(
                        roc_auc_score(y_test[:, i], y_pred[:, i])
                    )
                metrics_results['roc_auc'] = np.mean(roc_auc_scores)
            else:
                metrics_results['roc_auc'] = roc_auc_score(y_test, y_pred, average=self.f1_average, multi_class='ovr')

        return metrics_results

    def evaluate_to_dataframe(self, X_test, y_test):
        results = self.evaluate(X_test, y_test)
        print("Evaluation results:", results)  # Debugging line to inspect results
        df_results = pd.DataFrame(list(results.items()), columns=['Metric', 'Score'])
        return df_results
