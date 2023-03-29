from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
import numpy as np

final_data=pd.read_csv("output.csv")
y=final_data['Price'].to_numpy()
X=final_data.drop(['Price'],axis=1).to_numpy()
train_x,test_x,train_y,test_y=train_test_split(X,y,random_state=39,test_size=0.2)
model=KNeighborsRegressor(n_neighbors=20)
model.fit(train_x,train_y)
output=model.predict(test_x)
ans=np.sqrt(mean_squared_error(test_y,output))
print(ans)
pickle.dump(model,open("model.pkl","wb"))