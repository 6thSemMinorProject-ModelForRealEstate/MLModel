from sklearn.ensemble import RandomForestRegressor,VotingRegressor
from sklearn.svm import SVR
from lazypredict.Supervised import LazyRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
final_data=pd.read_csv("output.csv")
# print(final_data.head())
y=final_data['Price'].to_numpy()
X=final_data.drop(['Price'],axis=1)
train_x,test_x,train_y,test_y=train_test_split(X,y,random_state=45,test_size=0.2)
random=RandomForestRegressor(random_state=6,max_depth=10,n_estimators=200,min_samples_split=2,min_samples_leaf=1,max_features="log2",verbose=0)
svc=SVR(kernel="rbf",tol=0.001)
vote=VotingRegressor([("svc",svc),("random",random)])
vote.fit(train_x,train_y)
output=vote.predict(test_x)
ans=mean_squared_error(test_y,output)
print(ans)
print(test_x.shape)
print(y.shape)
pickle.dump(vote,open("model.pkl","wb"))