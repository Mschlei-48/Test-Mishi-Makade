import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics

data=pd.read_csv("tourism_data_500_points.csv")
print(data.head())
print(data.isnull().sum())
print(data.duplicated())
print(f"Shape of data before dropping suplicates: {data.shape}")
data=data.drop_duplicates()
print(f"Shape of data after dropping suplicates: {data.shape}")
data=pd.get_dummies(data,columns=["Date","Location"])
print(data.head())
print(data.shape)
print(data.info())

features=data.iloc[:,1:]
target=data.iloc[:,0]
x_train,x_test,y_train,y_test=train_test_split(features,target,test_size=0.3)
model=Lasso()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
r2scores=metrics.r2_score(y_test,y_pred)
mae=metrics.mean_absolute_error(y_test,y_pred)

print(f"R2-Score: {r2scores} \n MAE-Score: {mae}")

with open("results.txt","w") as f:
    f.write(f"R2-Score: {r2scores} \n MAE-Score: {mae}")


