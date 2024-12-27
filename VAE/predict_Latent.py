import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from VAE import VAE
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

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


def evaluate_model(model_normal, model_anomaly, test_data, true_labels, device, n=2):
    with torch.no_grad():
        # 获取正常和异常模型的重建和潜在表示
        reconstruction_normal, mu_normal, logvar_normal = model_normal(test_data)
        reconstruction_anomaly, mu_anomaly, logvar_anomaly = model_anomaly(test_data)

        # 计算重建误差
        error_normal = torch.mean((reconstruction_normal - test_data) ** 2, dim=2)
        error_anomaly = torch.mean((reconstruction_anomaly - test_data) ** 2, dim=2)

        # 重新调整潜在表示，应用缩放系数 n
        adjusted_mu_normal = mu_normal * (error_normal ** n)
        adjusted_mu_anomaly = mu_anomaly * (error_anomaly ** n)

        # 将规范聚合为每个示例的单个值
        aggregated_norm_normal = adjusted_mu_normal.mean(dim=1)
        aggregated_norm_anomaly = adjusted_mu_anomaly.mean(dim=1)

        # 预测标签：比较调整后的潜在表示的误差
        predictions = (aggregated_norm_anomaly < aggregated_norm_normal).cpu().numpy()

        # 性能评估
        precision = precision_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)

        # 混淆矩阵
        cm = confusion_matrix(true_labels, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

        # 可视化潜在空间
        plot_latent_space(mu_normal, mu_anomaly, "Original Latent Space")
        plot_latent_space(adjusted_mu_normal, adjusted_mu_anomaly, "Adjusted Latent Space")

        return precision, recall, f1


def plot_latent_space(mun, mua, title="Latent Space"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')  # 设置3D视图

    # 确保mu是在CPU上，并转换为numpy，因为matplotlib不接受GPU tensors
    mun = mun.cpu().numpy()
    mua = mua.cpu().numpy()

    # 画出每个点，假设mu已经是(n_samples, 3)的形状
    ax.scatter(mun[:, 0], mun[:, 1], mun[:, 2], alpha=0.7)
    ax.scatter(mua[:, 0], mua[:, 1], mua[:, 2], alpha=0.7)

    ax.set_xlabel('Latent Dimension 1')
    ax.set_ylabel('Latent Dimension 2')
    ax.set_zlabel('Latent Dimension 3')
    plt.title(title)
    plt.show()


test_data, true_labels = load_data('dataset/balanced_test_data.csv', device)
model_normal = load_model('data/normal.pth', device)
model_anomaly = load_model('data/anomaly.pth', device)
precision, recall, f1 = evaluate_model(model_normal, model_anomaly, test_data, true_labels, device)
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
