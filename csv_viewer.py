import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取两个CSV文件
file1 = 'algo_obs.csv'
file2 = 'recorddata/algo_obs.csv'

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# 检查两个DataFrame的列是否相同
if list(df1.columns) != list(df2.columns):
    raise ValueError("两个CSV文件的列不匹配")

# 绘制每个列的对比图
for column in df1.columns:
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df1, x=df1.index, y=column, label='File 1')
    sns.lineplot(data=df2, x=df2.index, y=column, label='File 2')
    plt.title(f'Comparison of {column}')
    plt.xlabel('Index')
    plt.ylabel(column)
    plt.legend()
    plt.show()