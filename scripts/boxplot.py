import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv("D:\项目分析\pythonProject\phenotype\data\test.csv")
print(df)
def boxplot(data):
    fig = sns.boxplot(x="ID", y="index", data=data)
    fig.show