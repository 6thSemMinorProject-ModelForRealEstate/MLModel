from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,BaggingRegressor
from sklearn.svm import SVR,NuSVR,LinearSVR
from sklearn.tree import DecisionTreeRegressor,ExtraTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

final_data=pd.read_csv("../MLdataset/final_dataset.csv")
final_data.corr().to_csv("feature_dependency_on_price.csv")
y=final_data['Price'].to_numpy()
X=final_data.drop(['Price'],axis=1)
scores=[]
points=[]

models=[LinearRegression(),LinearSVR(random_state=45),ExtraTreeRegressor(random_state=45),ExtraTreesRegressor(random_state=45),RandomForestRegressor(random_state=45),BaggingRegressor(random_state=45),SVR(),NuSVR(),DecisionTreeRegressor(random_state=45),KNeighborsRegressor(n_neighbors=17)]
# for i in range(1,100):
#     model=KNeighborsRegressor(n_neighbors=17)
    # train_x,test_x,train_y,test_y=train_test_split(X,y,test_size=0.2,random_state=i)
#     model.fit(train_x,train_y)
#     scores.append(pow(mean_squared_error(model.predict(test_x),test_y),0.5))
#     points.append(i)
for i in range(1,100):
    print(i)
    train_x,test_x,train_y,test_y=train_test_split(X,y,test_size=i/100,random_state=94)
    kNeighbourModel = BaggingRegressor(random_state=52,n_estimators=9)
    kNeighbourModel.fit(train_x, train_y)
    output=kNeighbourModel.predict(test_x)
    ans=np.sqrt(mean_squared_error(test_y,output))
    points.append(i)
    scores.append(ans)
import matplotlib.pyplot as plt
plt.xlabel("N-Estimator")
plt.ylabel("Error")
plt.title("N-Estimator Vs Error")
plt.subplot(3,1,1)
plt.bar(points,scores)
plt.subplot(3,1,2)
plt.plot(points,scores)
plt.subplot(3,1,3)
plt.scatter(points,scores)
# plt.xticks(rotation=90)
plt.show()
# data=pd.Series(scores,index=points)
# data.to_csv("data.csv")
