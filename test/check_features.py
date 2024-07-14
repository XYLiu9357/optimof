import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
# import seaborn as sns 

# sns.set_theme("paper", "dark")

print("***Entering check_features***")

file_path = "test/features.csv"
df = pd.read_csv(file_path)

print(df.describe())

print("***Exiting check_features***")