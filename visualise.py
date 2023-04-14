import matplotlib.pyplot as plt
import pickle
import pandas as pd
data=pd.read_csv("dataset.csv")
plt.xlabel("Room Size")
plt.ylabel("Property Value")
plt.title("Room Size Vs Property Value")
plt.scatter(data['room_size'], data['property_value'], marker='o')
plt.show()