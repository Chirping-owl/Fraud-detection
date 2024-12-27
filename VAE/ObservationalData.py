import pandas as pd

# 载入数据
data_path = './dataset/creditcard.csv'

raw_data = pd.read_csv(data_path)

# 2.1 训练数据集 - 快速概述
print(raw_data.head())

# 2.2 训练数据集 - 基本统计
print(raw_data.describe())
