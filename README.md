# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: 
RegisterNumber:  
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(x_test,y_test,color='blue')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
### Dataset
![output]![Screenshot 2024-02-27 175746](https://github.com/MOHAMEDFARIKH1/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/160568234/31317b6c-77da-46e8-913c-46c14d3c8e5e)

###df.head
![output]![Screenshot 2024-02-27 175327](https://github.com/MOHAMEDFARIKH1/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/160568234/f9500330-c2cb-4bc9-a574-de0974b8a53f)

###df.tail
![output]![Screenshot 2024-02-27 175334](https://github.com/MOHAMEDFARIKH1/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/160568234/70fb9e73-ecf3-4793-bb35-e3620603d5d3)

### X and Y values
![output]![Screenshot 2024-02-27 175346](https://github.com/MOHAMEDFARIKH1/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/160568234/10b4cade-4dd8-4f80-b6a3-a5ea612dc38b)

### Predication values of X and Y
![output]![Screenshot 2024-02-27 175355](https://github.com/MOHAMEDFARIKH1/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/160568234/d1db4c18-8c23-488a-a124-1807cd1f0de8)

### MSE,MAE and RMSE
![output]![Screenshot 2024-02-27 175405](https://github.com/MOHAMEDFARIKH1/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/160568234/2bb8a767-2018-4bd2-b6a0-ee0decbbda84)

### Training Set
![output]![Screenshot 2024-02-27 175541](https://github.com/MOHAMEDFARIKH1/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/160568234/a8e1d0da-307f-4ada-ae2f-e85cd1cd9143)

### Testing Set
![output]![Screenshot 2024-02-27 175617](https://github.com/MOHAMEDFARIKH1/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/160568234/5a41b3d9-dc27-44aa-8974-0388a5d35669)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
