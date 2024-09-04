import pandas as pd
import re
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

class BaseDatasetTransform:
    def __init__(self, dataset, target = None,dataset_test=None):
        self.dataset = dataset

        if isinstance(target, str):
            self.target = [target]  
        elif isinstance(target, list):
            self.target = target   
        else:
            raise ValueError("Target should be string or a list of strings")

        self.dataset_test = dataset_test

    def _remove_id_columns(self, X):

        cols = X.columns


        #id_columns = [col for col in cols if col.lower() == 'id']
        id_columns = [col for col in cols if 'id' in col.lower()]

        if id_columns:
            print(f'\nУдаление колонок с именем "id": {id_columns}')
            X = X.drop(columns=id_columns)
            print('-------------------------------------------')

        return X

    def fit_transform(self):

        self.dataset = self._remove_id_columns(self.dataset)
        if self.target:
            #if self.target in self.dataset.columns:
                #target_missing = self.dataset[self.target].isna().sum()
                target_missing = self.dataset[self.target].isna().sum(axis=0)

                target_missing_columns = target_missing[target_missing>0]
                print(target_missing_columns)
                if not target_missing_columns.empty:
                    for column in target_missing_columns:
                        print(f'\nВ колонке {column} есть пропущенные значения:')
                        print(f'Количество пропущенных значений: {target_missing[column]}')

                        self.dataset = self.dataset.dropna(subset=[column])
                else:
                    print(f'\nВ колонке(ах) {self.target} нет пропущенных значений')

        missing_values = self.dataset.isna().sum()
        missing_columns = missing_values[missing_values > 0]
        print('-------------------------------------------')

        if not missing_columns.empty:
            print('В наборе данных есть пропущенные значения:')
            print("Колонки с пропущенными значениями:")
            print(missing_columns)

            for col in missing_columns.index:
                if self.dataset[col].dtype in ['object','category']:
                    imputer = SimpleImputer(strategy='most_frequent')
                else:
                    imputer = SimpleImputer(strategy='median')

                self.dataset[col] = imputer.fit_transform(self.dataset[[col]]).ravel()

            missing_values_after = self.dataset.isna().sum()
            missing_columns_after = missing_values_after[missing_values_after > 0]
            if not missing_columns_after.empty:
                print('После заполнения все еще есть пропущенные значения:')
                print(missing_columns_after)
            else:
                print('Все пропущенные значения заполнены.')

            
        else:
            print('В наборе данных нет пропущенных значений')

        print('-------------------------------------------')
        print('Информация о колонках в датасете')
        categorical_cols = self.dataset.drop(self.target,axis=1).select_dtypes(include=['object', 'category']).columns
        numeric_cols = self.dataset.select_dtypes(include=['number']).columns
        
        print("\nКатегориальные колонки:")
        print(categorical_cols)

        print("\nЧисловые колонки:")
        print(numeric_cols)

        if self.dataset_test is not None:
          return self.dataset, categorical_cols, self.dataset_test

        return self.dataset, categorical_cols.to_list()

    def get_train_test_split(self, test_size=0.3, random_state=42):
        print('Количество значений целевой переменной по категориям:')
        
        # Проверяем, является ли target одиночной переменной или списком
        if len(self.target) == 1:
            # Одиночная целевая переменная
            y = self.dataset[self.target[0]]
            X = self.dataset.drop(columns=self.target)
            
            print(self.dataset[self.target[0]].value_counts())
            print('Следует ли выполнить стратифицированное раздеение на обучающую и тестовую выборку? y/n')
            stratify_split = input() == 'y'
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y if stratify_split else None
            )
            
        else:
            # Мультицелевые переменные
            y = self.dataset[self.target]
            X = self.dataset.drop(columns=self.target)
            
            # Стратификация для мультицелевых переменных
            # Объединяем значения целевых переменных в строку для стратификации
            y_stratify = y.apply(lambda row: '_'.join(row.astype(str)), axis=1)
            
            print('Количество значений комбинаций целевых переменных по категориям:')
            print(y_stratify.value_counts())
            print('Следует ли выполнить стратифицированное раздеение на обучающую и тестовую выборку? y/n')
            stratify_split = input() == 'y'
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y_stratify if stratify_split else None
            )

        print('Разделение датасета выполнено успешно')
        return X_train, X_test, y_train, y_test
