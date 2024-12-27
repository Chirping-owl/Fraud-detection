import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 加载数据集
data_0 = pd.read_csv('./dataset/creditcard_0.csv')
data_1 = pd.read_csv('./dataset/creditcard_1.csv')

# 列出数值型特征
numeric_columns = list(data_0.columns.drop('Class'))

# 创建绘图
fig = plt.figure(figsize=(20, 50))
rows, cols = 10, 3  # 调整为需要的行列数

for idx, num in enumerate(numeric_columns[:30]):  # 确保不超过30个特征，或调整为实际数量
    ax = fig.add_subplot(rows, cols, idx+1)
    ax.grid(alpha=0.7, axis="both")
    sns.kdeplot(x=num, fill=True, color="#3386FF", linewidth=0.6, data=data_0, label="Normal")
    sns.kdeplot(x=num, fill=True, color="#EFB000", linewidth=0.6, data=data_1, label="Anomaly")
    ax.set_xlabel(num)
    ax.legend()

fig.tight_layout()
plt.show()
