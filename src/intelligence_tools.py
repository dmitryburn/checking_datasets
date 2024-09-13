import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from scipy.stats import chi2_contingency
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


class AnalyzeVariables:
    def __init__(self,df:pd.DataFrame):
        self.df = df
   
    def info(self):
        return self.df.info()
    
    def shape(self):
        return f'{self.df.shape[0]} - строк и {self.df.shape[1]} колонок с переменными в датасете'
    
    def basic_statistics(self,columns=None):
        if columns:
            return self.df[columns].describe()
        else:
            return self.df.describe()
        
    def categorical_hist(self,cat_cols=None,bins=20):
        cat_cols = [col for col in self.df.columns if self.df[col].dtype in ['object', 'category']]
        return self.df[cat_cols].hist()
    
    def missing_percentage(self,columns:list=None):
        if columns:
            return 100 * self.df[columns].isna().sum()/self.df.shape[0]
        else:
            return 100 * self.df.isna().sum()/self.df.shape[0] 
        
    def missing_heatmap(self,columns:list=None):
        def plot_func(df):
            plt.figure(figsize=(30, 10))  
            sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
            plt.title('Тепловая карта пропущенных значений')
            plt.xlabel('Столбцы')
            plt.ylabel('Строки')
            plt.show()

        if columns and (len(columns) > 20) or len(self.df.columns) > 20 :
            raise ValueError('Too many columns (max 20). Bad visual. You can use missing_percentage() function as alternative')
        
        if columns:
            return plot_func(self.df[columns])
        else:
            return plot_func(self.df)
        
    def _get_num_cols(self) -> list:
        return [col for col in self.df.columns if self.df[col].dtype not in ['object', 'category']]

    def _get_cat_cols(self) -> list:
        return [col for col in self.df.columns if self.df[col].dtype in ['object', 'category']]

    def corr_matrix(self,columns=None):
        num_cols = self._get_num_cols()
        if columns and (columns not in num_cols):
            raise ValueError('You passed columns that is not numeric')
                    
        if columns:
            return self.df[columns].corr()
        else:
            return self.df[num_cols].corr()
    
    def corr_heatmap(self,columns=None):
        plt.figure(figsize=(8, 6))  
        sns.heatmap(self.corr_matrix(columns), annot=True, cmap="coolwarm", fmt='.2f', linewidths=0.5)
        plt.title("Корреляционная тепловая карта")
        plt.show()

    
    def feature_imortance(self,target):
        cat_cols = [col for col in self._get_cat_cols() if col not in target]
        num_cols = [col for col in self._get_num_cols() if col not in target]

        num_imputer = SimpleImputer(strategy='median')
        X_num = num_imputer.fit_transform(self.df[num_cols])

        scaler = StandardScaler()
        X_num_scaled = scaler.fit_transform(X_num)

        def cramers_v(x, y):
            confusion_matrix = pd.crosstab(x, y)
            chi2 = chi2_contingency(confusion_matrix)[0]
            n = confusion_matrix.sum().sum()
            r, k = confusion_matrix.shape
            return np.sqrt(chi2 / (n * (min(r, k) - 1)))
        
        cramers_v_value = []
        reg_coefs = []
        for t in target:
            model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
            model.fit(X_num_scaled, self.df[target])
            reg_coefs.append(model.coef_[0])

            temp = []
            for col in cat_cols:
                temp.append(cramers_v(self.df[col],self.df[t]))

            cramers_v_value.append(temp)
        
        categorical_importance = pd.DataFrame(cramers_v_value,columns=cat_cols,index=target)
        numeric_importance = pd.DataFrame(reg_coefs,columns=num_cols,index=target)


        return categorical_importance,numeric_importance
    
    def detect_outliers(self, columns=None):
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



        

        
        
        
                

        



        
    

        
        


    
        
        


    




    
    