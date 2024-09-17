import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from embeddings_tools import get_embeddings
from typing import List, Optional, Tuple, Union

class BaseDatasetTransform:
    """
    Класс для предварительной обработки набора данных, включая удаление идентификаторных столбцов, 
    заполнение пропущенных значений и преобразование текстовых данных в эмбеддинги.

    Параметры:
    - dataset: Исходный набор данных (pandas DataFrame).
    - text_columns: Список столбцов, содержащих текстовые данные, которые будут преобразованы в эмбеддинги. По умолчанию None.
    - target: Целевой признак (или признаки) для задачи. Может быть строкой (один признак) или списком строк (несколько признаков).
    - dataset_test: Дополнительный набор данных для тестирования (pandas DataFrame). По умолчанию None.
    - multilanguage: Флаг для поддержки нескольких языков в функции get_embeddings. По умолчанию None.
    """
    
    def __init__(self, dataset: pd.DataFrame, text_columns: Optional[List[str]] = None,
                 target: Optional[Union[str, List[str]]] = None, dataset_test: Optional[pd.DataFrame] = None,
                 multilanguage: Optional[bool] = None):
        """
        Инициализирует класс BaseDatasetTransform.

        :param dataset: Исходный набор данных (pandas DataFrame).
        :param text_columns: Список столбцов с текстовыми данными. По умолчанию None.
        :param target: Целевой признак (или признаки). Может быть строкой или списком строк.
        :param dataset_test: Дополнительный набор данных для тестирования. По умолчанию None.
        :param multilanguage: Флаг для поддержки многократных языков в функции get_embeddings. По умолчанию None.
        """
        self.dataset = dataset
        self.text_columns = text_columns
        self.multilanguage = multilanguage

        if isinstance(target, str):
            self.target = [target]
        elif isinstance(target, list):
            self.target = target
        else:
            raise ValueError("Target should be string or a list of strings")
        
        self.dataset_test = dataset_test

    def _remove_id_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Удаляет столбцы, содержащие 'id' в названии, из набора данных.

        :param X: Набор данных (pandas DataFrame).
        :return: Обновленный набор данных с удаленными столбцами.
        """
        cols = X.columns
        id_columns = [col for col in cols if 'id' in col.lower()]
        if id_columns:
            print(f'\nОбнаружены колонки с именем "id": {id_columns}')
            for col in id_columns:
                response = input(f'Хотите удалить колонку "{col}"? (y/n): ').strip().lower()
                if response == 'y':
                    X = X.drop(columns=[col])
                    print(f'Колонка "{col}" удалена.')
        return X

    def _impute_missing_values(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """
        Заполняет пропущенные значения в обучающем и тестовом наборах данных.

        :param X_train: Обучающий набор данных (pandas DataFrame).
        :param X_test: Тестовый набор данных (pandas DataFrame).
        :return: Кортеж из обновленного обучающего набора данных, обновленного тестового набора данных и списка категориальных колонок.
        """
        missing_columns = X_train.isna().sum()
        missing_columns = missing_columns[missing_columns > 0]
        
        if not missing_columns.empty:
            print('В наборе данных есть пропущенные значения:')
            print("Колонки с пропущенными значениями:")
            print(missing_columns)

            for col in missing_columns.index:
                if X_train[col].dtype in ['object', 'category']:
                    imputer = SimpleImputer(strategy='most_frequent')
                else:
                    imputer = SimpleImputer(strategy='median')
                
                X_train[col] = imputer.fit_transform(X_train[[col]]).ravel()
                X_test[col] = imputer.transform(X_test[[col]]).ravel()

            missing_values_after = X_train.isna().sum()
            missing_values_after = missing_values_after[missing_values_after > 0]
            if not missing_values_after.empty:
                print('После заполнения все еще есть пропущенные значения в обучающем наборе:')
                print(missing_values_after)
            else:
                print('Все пропущенные значения в обучающем наборе заполнены.')
            
        else:
            print('В наборе данных нет пропущенных значений')

        print('-------------------------------------------')
        print('Информация о колонках в датасете')
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
        numeric_cols = X_train.select_dtypes(include=['number']).columns
        
        print("\nКатегориальные колонки:")
        print(categorical_cols)

        print("\nЧисловые колонки:")
        print(numeric_cols)

        return X_train, X_test, categorical_cols.to_list()

    def fit_transform(self, test_size: float = 0.3) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Optional[pd.DataFrame], List[str]]:
        """
        Выполняет предварительную обработку данных, включая удаление столбцов, заполнение пропусков, 
        преобразование текста в эмбеддинги и разделение данных на обучающий и тестовый наборы.

        :param test_size: Размер тестовой выборки. Доля от общего числа данных. По умолчанию 0.3.
        :return: Кортеж из обучающего набора данных, тестового набора данных, целевых переменных для обучения,
                 целевых переменных для тестирования, дополнительного набора данных для тестирования (если указан), 
                 и списка категориальных колонок.
        """
        self.dataset = self._remove_id_columns(self.dataset)
        
        if self.target:
            target_missing = self.dataset[self.target].isna().sum(axis=0)
            target_missing_columns = target_missing[target_missing > 0]
            print(target_missing_columns)
            if not target_missing_columns.empty:
                for column in target_missing_columns:
                    print(f'\nВ колонке {column} есть пропущенные значения:')
                    print(f'Количество пропущенных значений: {target_missing[column]}')
                    self.dataset = self.dataset.dropna(subset=[column])
            else:
                print(f'\nВ колонке(ах) {self.target} нет пропущенных значений')
                
        if self.text_columns: 
            self.dataset = get_embeddings(self.dataset, self.text_columns, batch_size=32, multilanguage=self.multilanguage)

        X = self.dataset.drop(columns=self.target)
        y = self.dataset[self.target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y if len(self.target) == 1 else None
        )
        
        X_train, X_test, categorical_cols = self._impute_missing_values(X_train, X_test)

        if self.dataset_test is not None:
            self.dataset_test = self._remove_id_columns(self.dataset_test)
            X_test = self._impute_missing_values(X_train, self.dataset_test)[1]
            return X_train, X_test, y_train, y_test, self.dataset_test, categorical_cols

        return X_train, X_test, y_train, y_test, categorical_cols
