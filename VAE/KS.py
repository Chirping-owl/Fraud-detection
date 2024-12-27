import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

# 加载数据
data_path = './dataset/creditcard.csv'  # 数据集路径

raw_data = pd.read_csv(data_path)

# 2. 双侧 Kolmogorov-Smirnov (KS) 检验
# 假设'Class'是目标列，1表示异常，0表示正常
normal_data = raw_data[raw_data['Class'] == 0]
anomaly_data = raw_data[raw_data['Class'] == 1]

# 对每个特征进行KS检验
ks_results = {}
for column in raw_data.columns.drop(['Class']):
    ks_stat, ks_pvalue = stats.ks_2samp(normal_data[column], anomaly_data[column])
    ks_results[column] = (ks_stat, ks_pvalue)


# 打印KS检验结果
ks_results_df = pd.DataFrame(ks_results, index=['KS Statistic', 'P-Value']).T
print(ks_results_df.sort_values(by='P-Value'))

# 根据P-Value的值进行筛选
filtered_results = ks_results_df[ks_results_df['P-Value'] > 1.5e-07]

# 打印筛选后的结果，按P-Value排序
print(filtered_results)