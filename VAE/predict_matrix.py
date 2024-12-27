import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from VAE import VAE
import seaborn as sns

# 确定使用的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(file_path, device):
    data = pd.read_csv(file_path)
    true_labels = data['Class'].values
    data = data.drop(columns='Class')
    data_tensor = torch.tensor(data.values, dtype=torch.float32).reshape(-1, 1, 23)
    data_tensor = data_tensor.to(device)  # 移动数据到指定设备
    return data_tensor, true_labels

def load_model(model_path, device):
    model = torch.load(model_path)
    model = model.to(device)  # 确保模型在正确的设备上
    model.eval()
    return model


def evaluate_model(model_normal, model_anomaly, test_data, true_labels, device):
    with torch.no_grad():
        reconstruction_normal, _, _ = model_normal(test_data)
        reconstruction_anomaly, _, _ = model_anomaly(test_data)
        error_normal = torch.mean((reconstruction_normal - test_data) ** 2, dim=2)
        error_anomaly = torch.mean((reconstruction_anomaly - test_data) ** 2, dim=2)

        # 将误差张量转移到CPU并转换为NumPy数组
        error_normal = error_normal.cpu().numpy()
        error_anomaly = error_anomaly.cpu().numpy()

        # 预测标签：如果异常重建误差小于正常重建误差，则预测为异常
        predictions = (error_anomaly < error_normal)

        # 计算性能指标
        precision = precision_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)

        # 绘制混淆矩阵
        cm = confusion_matrix(true_labels, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

        return precision, recall, f1


test_data, true_labels = load_data('dataset/balanced_test_data.csv', device)
model_normal = load_model('data/normal.pth', device)
model_anomaly = load_model('data/anomaly.pth', device)
precision, recall, f1 = evaluate_model(model_normal, model_anomaly, test_data, true_labels, device)
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

