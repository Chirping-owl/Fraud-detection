import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

# 加载数据
data_paths = ['./dataset/creditcard_0.csv','./dataset/creditcard_1.csv']  # 数据集路径

for data_path in data_paths:
    raw_data = pd.read_csv(data_path)

    # 1. 特征共线性检验
    # 计算相关系数矩阵
    correlation_matrix = raw_data.drop(['Class'], axis=1).corr()  # 'Class'是目标列

    # 绘制热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, linewidths=0.5, linecolor='white')
    plt.title('Feature Correlation Matrix')
    plt.show()
