'''
the project on House price prediction
'''
import pandas as pd
import numpy as np
import sys
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
class Data:
    def __init__(self, data):
        try:
            self.df = pd.read_csv(data)
            self.df['Neighborhood']=self.df['Neighborhood'].map({'Rural':0,'Suburb':1,'Urban':2}).astype(int)
            self.x = self.df.iloc[:, :-1]
            self.y = self.df.iloc[:, -1]
            self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(self.x,self.y,test_size=0.1,random_state=42)
        except Exception as e:
            error_msg, error_line, error_type = sys.exc_info()
            print(f'error_line->{error_line.tb_lineno}, error_msg->{error_msg}, error_type->{error_type}')

    def LASSO(self):
      try:
        self.reg1 = Lasso(alpha=0.5)
        self.reg1.fit(self.x_train,self.y_train)
        self.y_train_pred = self.reg1.predict(self.x_train)
        self.reg1.fit(self.x_test,self.y_test)
        self.y_test_pred = self.reg1.predict(self.x_test)
        print("R2_score value = ",r2_score(self.y_train,self.y_train_pred))
        print("mean_squared_error =",mean_squared_error(self.y_train,self.y_train_pred))
        print("R2_score value = ",r2_score(self.y_test,self.y_test_pred))
        print("mean_squared_error =",mean_squared_error(self.y_test,self.y_test_pred))
      except Exception as e:
        error_msg, error_line, error_type = sys.exc_info()
        print(f'error_line->{error_line.tb_lineno}, error_msg->{error_msg}, error_type->{error_type}')

    def RIDGE(self):
      try:
        self.reg2 =Ridge(alpha=0.5)
        self.reg2.fit(self.x_train,self.y_train)
        y_train_pred = self.reg2.predict(self.x_train)
        self.reg2.fit(self.x_test,self.y_test)
        self.y_test_pred = self.reg2.predict(self.x_test)
        print("R2_score value = ",r2_score(self.y_train,self.y_train_pred))
        print("mean_squared_error =",mean_squared_error(self.y_train,self.y_train_pred))
        print("R2_score value = ",r2_score(self.y_test,self.y_test_pred))
        print("mean_squared_error =",mean_squared_error(self.y_test,self.y_test_pred))
      except Exception as e:
        error_msg, error_line, error_type = sys.exc_info()
        print(f'error_line->{error_line.tb_lineno}, error_msg->{error_msg}, error_type->{error_type}')
if __name__ == "__main__":
     try:
        obj = Data('C:\\Users\\abc\\Downloads\\ML\\pythonProject\\housing_price_dataset.csv')
        obj.LASSO()
        obj.RIDGE()
     except Exception as e:
        error_msg, error_line, error_type = sys.exc_info()
        print(f'error_line-> {error_type.tb_lineno} error_msg->{error_msg}, error_type->{error_type}')


