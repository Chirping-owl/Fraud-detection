import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support
from VAE import VAE  # 从VAE.py导入VAE类
import os

# 设置使用的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 加载数据的函数
def load_data(file_path, has_labels=False, batch_size=64):
    data = pd.read_csv(file_path)
    labels = data['Class'].values
    data = data.drop(columns='Class')
    data_tensor = torch.tensor(data.values, dtype=torch.float32).reshape(-1, 1, 23)
    dataset = TensorDataset(data_tensor, torch.tensor(labels)) if has_labels else TensorDataset(data_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# 模型加载函数
def load_model(model_path):
    model = torch.load(model_path)
    model.eval()
    model.to(device)
    return model


# 计算重建误差的函数
def calculate_reconstruction_error(model, dataloader):
    reconstruction_errors = []
    with torch.no_grad():
        for data in dataloader:
            inputs = data[0].to(device)
            reconstruction, _, _ = model(inputs)
            mse = torch.mean((inputs - reconstruction) ** 2, dim=[1, 2])
            reconstruction_errors.extend(mse.cpu().numpy())
    return np.array(reconstruction_errors)


# 确定阈值的函数
def determine_threshold(errors, quantile=0.3):
    return np.quantile(errors, quantile)


# 模型评估函数
def evaluate_model(model, dataloader, threshold):
    true_labels = []
    predictions = []
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data[0].to(device), data[1]
            reconstruction, _, _ = model(inputs)
            mse = torch.mean((inputs - reconstruction) ** 2, dim=[1, 2])
            preds = (mse > threshold).long()
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.numpy())
    f1 = f1_score(true_labels, predictions)
    return f1


# 加载模型
model_normal = load_model('data/normal.pth')

# 加载训练数据
train_loader = load_data('dataset/normal_train.csv', batch_size=128)

# 计算重建误差
reconstruction_errors = calculate_reconstruction_error(model_normal, train_loader)

# 确定异常检测阈值
threshold = determine_threshold(reconstruction_errors)

# 加载测试数据
test_loader = load_data('dataset/balanced_test_data.csv', has_labels=True, batch_size=128)

# 评估模型
f1_score = evaluate_model(model_normal, test_loader, threshold)
print(f"F1 Score: {f1_score}")
