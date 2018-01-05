
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm



dataset = pd.read_csv("criminal_train.csv")
test_dataset = pd.read_csv("criminal_test.csv")

x_dataset = dataset.loc[:,"IFATHER":"VEREP"]
y_dataset = dataset.loc[:,"Criminal":]

x_test_dataset = test_dataset.loc[:,"IFATHER":"VEREP"]
PERID_test_dataset = test_dataset.loc[:,"PERID":"PERID"]

svc = svm.SVC()
svc.fit(x_dataset,y_dataset.values.ravel())
pred = lr.predict(x_test_dataset)
pred = {'Criminal':pred}
pred = pd.DataFrame(pred)
result = pd.concat([PERID_test_dataset,pred], axis=1)
result.to_csv("submission.csv",index=False)
print(pred.shape,PERID_test_dataset.shape,result.shape)
pred.head(5)
PERID_test_dataset.head(5)
