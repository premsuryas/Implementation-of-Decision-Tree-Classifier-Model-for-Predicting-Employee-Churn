# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.USE IMPORT 
2. USE PANDAS
3. USE DATA
4. USE CSV

## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by:S.PREM KUMAR 
RegisterNumber:23013598  
*/
```
import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()
data.info()

data.isnull().sum()

data["left"].value_counts

from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x= data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]

x.head()
y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 100)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)

y_pred = dt.predict(x_test)
from sklearn import metrics

accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## Output:
!(![r1](https://github.com/premsuryas/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147473858/08d65bf8-6116-497d-9bcb-0c5f1a569765)
)
!(![r2](https://github.com/premsuryas/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147473858/65061ffd-baa8-4001-818c-0165b8b87a0d)
)
!(![r3](https://github.com/premsuryas/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147473858/0dba8525-d6cb-41f5-b65c-9738dda15611)
)
!(![r4](https://github.com/premsuryas/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147473858/f94ea84d-2c6d-41c8-ae58-3c721af59ef3)
)
!(![r5](https://github.com/premsuryas/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147473858/0ff113d8-cb1c-4151-b14b-e752d1449dd5)
)
!(![r6](https://github.com/premsuryas/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147473858/2203882d-6883-40c9-a37e-74b6af4a6948)
)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
