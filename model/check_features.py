import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
# import seaborn as sns 

# sns.set_theme("paper", "dark")

print("***Entering check_features***")

file_path = "model/features.csv"
df = pd.read_csv(file_path)

print(df.iloc[:, 3:5])
print(len(df.index))

print("***Exiting check_features***")