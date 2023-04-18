import matplotlib.pyplot as plt
import pickle
import pandas as pd
final_data=pd.read_csv("../MLdataset/final_dataset.csv")


plt.xlabel("Area")
plt.ylabel("City")
plt.title("Area Vs City")
plt.scatter(final_data['Area'], final_data['City'], marker='o')
plt.show()