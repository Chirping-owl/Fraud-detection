import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from tqdm import tqdm
from VAE import VAE  # 从VAE.py导入VAE类
import os

# 设置训练参数
epochs = 1000
lr = 1e-9
task = 'anomaly'  # 可以选择 'normal' 或 'anomaly'
# 确定使用的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 4
weight_decay = 5e-10
# 引入超参数β来平衡KL散度的影响
beta = 0.5  # β值可以根据实验调整，以找到最佳性能

# 数据加载函数
def load_data(file_path, batch_size=64):
    # 读取CSV文件
    data = pd.read_csv(file_path)
    data = data.drop(columns='Class')
    # 将数据转换为张量，修改形状以匹配模型输入 [batch_size, 1, 23]
    data_tensor = torch.tensor(data.values, dtype=torch.float32).reshape(-1, 1, 23)
    # 创建数据集和数据加载器
    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


# 根据任务类型选择数据文件
file_path = f'dataset/{task}_train.csv'
train_loader = load_data(file_path, batch_size)


# 权重初始化函数
def weight_init(m):
    if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d)):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


# 模型加载函数
def load_model(task, lr):
    model_path = f'model/{task}.pth'
    if os.path.exists(model_path):
        model = torch.load(model_path)
    else:
        model = VAE()
        # 应用权重初始化
        model.apply(weight_init)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return model, optimizer


# 加载模型和优化器
model, optimizer = load_model(task, lr)
model.to(device)  # 将模型移到正确的设备


# 训练函数
def train(model, train_loader, optimizer, epochs, device, clip_value=1.0):
    print(model)
    # 初始化调度器
    model.train()
    for epoch in range(epochs):
        with tqdm(total=len(train_loader.dataset), desc=f'Epoch {epoch + 1}/{epochs}', unit='条') as pbar:
            for data in train_loader:
                data = data[0].to(device)  # 将数据移到设备
                optimizer.zero_grad()
                reconstruction, mu, log_var = model(data)
                # 计算均方误差作为重构误差
                recon_error = torch.sqrt(torch.mean((reconstruction - data) ** 2))
                # 计算KL散度
                kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                loss = recon_error + beta * kld
                # 反向传播
                loss.backward()
                # 应用梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                optimizer.step()
                pbar.update(data.shape[0])
                pbar.set_postfix(loss=loss.item(), recon_error=recon_error.item(), kld=kld.item())

    # 保存模型
    torch.save(model, f'model/{task}.pth')


# 开始训练
train(model, train_loader, optimizer, epochs, device)
