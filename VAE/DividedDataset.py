import pandas as pd
from sklearn.model_selection import train_test_split

# 数据集路径
data_path = './dataset/creditcard.csv'

# 读取数据集
data = pd.read_csv(data_path)

# 分离异常和正常交易
anomalies = data[data['Class'] == 1]
normals = data[data['Class'] == 0]

# 删除指定的列
columns_to_drop = ['V13', 'V15', 'V22', 'V24', 'V25', 'V26', 'Time',]
data = data.drop(columns=columns_to_drop)

# 如果需要单独对正常和异常数据进行列删除
anomalies = anomalies.drop(columns=columns_to_drop)
normals = normals.drop(columns=columns_to_drop)



# 从异常交易中抽取100条作为训练VAE-anomaly模型的数据
anomaly_train, anomaly_test = train_test_split(anomalies, train_size=100, random_state=42)

# 从正常交易中抽取392条，与剩余的392条异常交易组合，形成评估用的平衡数据集
normal_for_test, _ = train_test_split(normals, train_size=392, random_state=42)
balanced_test_data = pd.concat([normal_for_test, anomaly_test])

# 剩余的正常交易数据用于训练VAE-normal模型
normal_train = normals.loc[normals.index.difference(normal_for_test.index)]

# 保存数据
anomaly_train.to_csv('./dataset/anomaly_train.csv', index=False)
balanced_test_data.to_csv('./dataset/balanced_test_data.csv', index=False)
normal_train.to_csv('./dataset/normal_train.csv', index=False)

print("数据划分完成，并保存到指定文件。")
