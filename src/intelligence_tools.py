import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from scipy.stats import chi2_contingency
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import List, Optional, Tuple, Dict


class AnalyzeVariables:
    """
    Класс для анализа переменных в наборе данных.

    Параметры:
    - df: Набор данных в формате pandas DataFrame.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Инициализирует класс AnalyzeVariables.

        :param 
        df: Набор данных (pandas DataFrame).
        target: Целевые колонки
        """
        self.df = df

    def info(self) -> None:
        """
        Выводит информацию о наборе данных.
        """
        return self.df.info()

    def shape(self) -> str:
        """
        Возвращает строку с информацией о размере набора данных.

        :return: Строка с количеством строк и колонок.
        """
        return f'{self.df.shape[0]} - строк и {self.df.shape[1]} колонок с переменными в датасете'

    def basic_statistics(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Возвращает основные статистические характеристики для указанных столбцов.

        :param columns: Список имен столбцов. Если None, возвращаются статистики для всех столбцов.
        :return: pandas DataFrame с основными статистиками.
        """
        if columns:
            return self.df[columns].describe()
        else:
            return self.df.describe()

    def categorical_hist(self, cat_cols: Optional[List[str]] = None, bins: int = 20) -> None:
        """
        Строит гистограммы для категориальных столбцов.

        :param cat_cols: Список категориальных столбцов. Если None, выбираются все категориальные столбцы.
        :param bins: Количество бинов для гистограмм.
        """
        cat_cols = cat_cols or [col for col in self.df.columns if self.df[col].dtype in ['object', 'category']]
        return self.df[cat_cols].hist(bins=bins)

    def missing_percentage(self, columns: Optional[List[str]] = None) -> pd.Series:
        """
        Возвращает процент пропущенных значений для указанных столбцов.

        :param columns: Список имен столбцов. Если None, возвращаются пропуски для всех столбцов.
        :return: pandas Series с процентом пропущенных значений.
        """
        if columns:
            return 100 * self.df[columns].isna().sum() / self.df.shape[0]
        else:
            return 100 * self.df.isna().sum() / self.df.shape[0]

    def missing_heatmap(self, columns: Optional[List[str]] = None) -> None:
        """
        Строит тепловую карту пропущенных значений.

        :param columns: Список столбцов для отображения. Если None, отображаются все столбцы.
        """
        def plot_func(df: pd.DataFrame) -> None:
            plt.figure(figsize=(30, 10))
            sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
            plt.title('Тепловая карта пропущенных значений')
            plt.xlabel('Столбцы')
            plt.ylabel('Строки')
            plt.show()

        if columns and (len(columns) > 20 or len(self.df.columns) > 20):
            raise ValueError('Слишком много столбцов (макс 20). Плохая визуализация. Используйте функцию missing_percentage() как альтернативу.')
        
        if columns:
            return plot_func(self.df[columns])
        else:
            return plot_func(self.df)

    def _get_num_cols(self) -> List[str]:
        """
        Возвращает список числовых столбцов в наборе данных.

        :return: Список имен числовых столбцов.
        """
        return [col for col in self.df.columns if self.df[col].dtype not in ['object', 'category']]

    def _get_cat_cols(self) -> List[str]:
        """
        Возвращает список категориальных столбцов в наборе данных.

        :return: Список имен категориальных столбцов.
        """
        return [col for col in self.df.columns if self.df[col].dtype in ['object', 'category']]

    def corr_matrix(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Возвращает корреляционную матрицу для указанных числовых столбцов.

        :param columns: Список имен числовых столбцов. Если None, возвращается корреляционная матрица для всех числовых столбцов.
        :return: pandas DataFrame с корреляционной матрицей.
        """
        num_cols = self._get_num_cols()
        if columns and any(col not in num_cols for col in columns):
            raise ValueError('Вы указали столбцы, которые не являются числовыми.')
                    
        if columns:
            return self.df[columns].corr()
        else:
            return self.df[num_cols].corr()

    def corr_heatmap(self, columns: Optional[List[str]] = None) -> None:
        """
        Строит тепловую карту корреляций.

        :param columns: Список имен числовых столбцов для отображения. Если None, отображается корреляция для всех числовых столбцов.
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.corr_matrix(columns), annot=True, cmap="coolwarm", fmt='.2f', linewidths=0.5)
        plt.title("Корреляционная тепловая карта")
        plt.show()

    def feature_importance(self, target: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Оценивает важность признаков для целевых переменных с помощью логистической регрессии и критерия Крамера.

        :param target: Список целевых переменных.
        :return: Кортеж из двух DataFrame: важность категориальных признаков и важность числовых признаков.
        """
        cat_cols = [col for col in self._get_cat_cols() if col not in target]
        num_cols = [col for col in self._get_num_cols() if col not in target]

        num_imputer = SimpleImputer(strategy='median')
        X_num = num_imputer.fit_transform(self.df[num_cols])

        scaler = StandardScaler()
        X_num_scaled = scaler.fit_transform(X_num)

        def cramers_v(x: pd.Series, y: pd.Series) -> float:
            """
            Вычисляет значение критерия Крамера для двух категориальных переменных.

            :param x: Первая категориальная переменная.
            :param y: Вторая категориальная переменная.
            :return: Значение критерия Крамера.
            """
            confusion_matrix = pd.crosstab(x, y)
            chi2 = chi2_contingency(confusion_matrix)[0]
            n = confusion_matrix.sum().sum()
            r, k = confusion_matrix.shape
            return np.sqrt(chi2 / (n * (min(r, k) - 1)))

        cramers_v_value = []
        reg_coefs = []
        for t in target:
            model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
            model.fit(X_num_scaled, self.df[t])
            reg_coefs.append(model.coef_[0])

            temp = [cramers_v(self.df[col], self.df[t]) for col in cat_cols]
            cramers_v_value.append(temp)

        categorical_importance = pd.DataFrame(cramers_v_value, columns=cat_cols, index=target)
        numeric_importance = pd.DataFrame(reg_coefs, columns=num_cols, index=target)

        return categorical_importance, numeric_importance

    def detect_outliers(self, columns: Optional[List[str]] = None) -> Dict[str, int]:
        """
        Обнаруживает выбросы в указанных числовых столбцах.

        :param columns: Список числовых столбцов для проверки на выбросы. Если None, проверяются все числовые столбцы.
        :return: Словарь, где ключи - имена столбцов, значения - количество выбросов в каждом столбце.
        """
        if not columns:
            columns = self._get_num_cols()

        outlier_counts = {}

        for col in columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_condition = (self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR))
            if outlier_condition.any(): 
                outlier_counts[col] = outlier_condition.sum()

        return outlier_counts
    
    def frequency_distribution(self,column:str) -> pd.DataFrame:
        """
        Вычисляет частотное распределение значений в указанной колонке DataFrame.

        Args:
            column (str): Имя колонки, для которой нужно вычислить частотное распределение.

        Returns:
            pd.DataFrame: DataFrame с частотным распределением.
                - Индекс: Уникальные значения в колонке.
                - 'counts': Количество вхождений каждого значения.
                - 'percentage_of_counts': Процентное соотношение вхождений каждого значения.
        """

        print(f'Частотное рапредедение переменной : {column}')
        val_counts = self.df[column].value_counts()

        df_counts = pd.DataFrame()

        df_counts['counts'] = val_counts.values
        df_counts['percentage_of_counts'] = np.round(100 * val_counts.values/np.sum(val_counts.values))
        df_counts.index = val_counts.index

        print(f'всего в колонке {column} имеется {len(val_counts.index)} уникальных занчений')
        return df_counts
    
    def get_column_type(self, column_name:str):
        """
        Определяет тип данных в столбце DataFrame.

        Args:
            column_name (str): Имя колонки.

        Returns:
            str: Тип данных в колонке.
        """

        column_data = self.df[column_name]

        if pd.api.types.is_numeric_dtype(column_data):
            if pd.api.types.is_integer_dtype(column_data):
                return 'integer'
            else:
                return 'float'
        elif pd.api.types.is_datetime64_dtype(column_data):
            return 'datetime'
        elif pd.api.types.is_object_dtype(column_data):
            return 'object'
        elif pd.api.types.is_bool_dtype(column_data):
            return 'bool'
        else:
            return 'unknown'
    
    def change_column_type(self, column:str, target_type):
        """
        Преобразует тип колонки в DataFrame.

        Args:
            column (str): Имя колонки для преобразования.
            target_type (str): Целевой тип данных. Допустимые значения:
                'int': integer
                'float': float
                'str': string
                'datetime': datetime
                'bool': boolean

        Returns:
            DataFrameModifier: Объект класса DataFrameModifier с измененным типом колонки.
        """
        if target_type == 'int':
            self.df[column] = pd.to_numeric(self.df[column], errors='coerce').astype(int)
        elif target_type == 'float':
            self.df[column] = pd.to_numeric(self.df[column], errors='coerce').astype(float)
        elif target_type == 'str':
            self.df[column] = self.df[column].astype(str)
        elif target_type == 'datetime':
            self.df[column] = pd.to_datetime(self.df[column], errors='coerce')
        elif target_type == 'bool':
            self.df[column] = self.df[column].astype(bool)
        else:
            raise ValueError(f'Некорректный тип данных: {target_type}. Допустимые значения: int, float, str, datetime, bool')

        return self