from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor
import numpy as np
import pickle

final_data=pd.read_csv("../MLdataset/final_dataset.csv")
y=final_data['Price'].to_numpy()
X=final_data.drop(['Price'],axis=1).to_numpy()
train_x,test_x,train_y,test_y=train_test_split(X,y,test_size=0.2,random_state=94)
kNeighbourModel = BaggingRegressor(random_state=52,n_estimators=9)
kNeighbourModel.fit(train_x, train_y)
output=kNeighbourModel.predict(test_x)
ans=np.sqrt(mean_squared_error(test_y,output))
print(ans)
pickle.dump(kNeighbourModel,open("model.pkl","wb"))