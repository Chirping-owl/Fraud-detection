import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from tqdm import tqdm
from VAE import VAE  # 确保VAE类已正确定义
import os

# 设置训练参数
epochs = 100
lr = 1e-5
batch_size = 64
weight_decay = 5e-8

# 确定使用的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 数据加载函数
def load_data(file_path, batch_size=64):
    data = pd.read_csv(file_path)
    data = data.drop(columns='Class')  # 假设数据集中'Class'是标签列
    data_tensor = torch.tensor(data.values, dtype=torch.float32).reshape(-1, 1, 23)
    dataset = TensorDataset(data_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# 权重初始化函数
def weight_init(m):
    if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d)):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


# 加载模型
def load_model(model_path, init_weights=True):
    if os.path.exists(model_path):
        model = torch.load(model_path)
    else:
        model = VAE()
        if init_weights:
            model.apply(weight_init)
    model.to(device)
    return model


# 准备训练数据
normal_loader = load_data('dataset/normal_train.csv', batch_size)
anomaly_loader = load_data('dataset/anomaly_train.csv', batch_size)

# 加载模型和优化器
normal_model = load_model('model/normal2.pth')
anomaly_model = load_model('model/anomaly2.pth')

optimizer_normal = optim.Adam(normal_model.parameters(), lr=lr, weight_decay=weight_decay)
optimizer_anomaly = optim.Adam(anomaly_model.parameters(), lr=lr, weight_decay=weight_decay)


# 训练函数
def train(tasks, models, loaders, optimizers, epochs, device, clip_value=1.0):
    for task, model, loader, optimizer in zip(tasks, models, loaders, optimizers):
        model.train()
        for epoch in range(epochs):
            with tqdm(total=len(loader.dataset), desc=f'Epoch {epoch + 1}/{epochs}', unit='sample') as pbar:
                for data in loader:
                    data = data[0].to(device)
                    optimizer.zero_grad()
                    reconstruction, mu, log_var = model(data)
                    recon_error = torch.mean((reconstruction - data) ** 2)
                    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                    loss = recon_error + kld
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                    optimizer.step()
                    pbar.update(data.shape[0])
                    pbar.set_postfix(loss=loss.item(), recon_error=recon_error.item(), kld=kld.item())
            torch.save(model, f'model/{task}2.pth')


# 执行训练
train(['normal', 'anomaly'], [normal_model, anomaly_model], [normal_loader, anomaly_loader],
      [optimizer_normal, optimizer_anomaly], epochs, device)
