import pandas as pd

# 数据集路径
data_path = './dataset/creditcard.csv'

# 读取数据集
data = pd.read_csv(data_path)

# 分割数据集基于'Class'列
data_1 = data[data['Class'] == 1]
data_0 = data[data['Class'] == 0]

# 保存到新的CSV文件
data_1.to_csv('./dataset/creditcard_1.csv', index=False)
data_0.to_csv('./dataset/creditcard_0.csv', index=False)

print("数据已成功分割并保存到 './dataset/creditcard_1.csv' 和 './dataset/creditcard_0.csv'")
