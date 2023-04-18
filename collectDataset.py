import os
import pandas as pd
import numpy as np

all_dataset=os.listdir("../MLdataset")
dfs=[]
count = 0
for i in range(len(all_dataset)):
    if(all_dataset[i]==".git"):continue
    df=pd.read_csv(f"../MLdataset/{all_dataset[i]}")
    df['City']=count
    df['City_name']=all_dataset[i].split(".")[0]
    df.dropna(inplace=True)
    df.drop_duplicates(keep="first",inplace=True)
    dfs.append(df)
    count+=1

df=pd.concat(dfs).reset_index().drop(['index'],axis=1)

df.to_csv("../MLdataset/collectDataset.csv",index=False)
