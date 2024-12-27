import torch
import torch.nn as nn


class SamplingLayer(nn.Module):
    """
    使用均值和对数方差进行采样的层，属于VAE模型的一部分。
    """

    def __init__(self):
        super(SamplingLayer, self).__init__()

    def forward(self, mu, log_var):
        std = torch.exp(0.5 * log_var)  # 标准差是对数方差的指数函数的一半
        eps = torch.randn_like(std)  # 从标准正态分布中采样
        return mu + eps * std  # 通过重参数化技巧返回采样结果


class Cropping1D(nn.Module):
    def __init__(self, crop_size):
        super(Cropping1D, self).__init__()
        self.crop_size = crop_size

    def forward(self, x):
        return x[:, :, :self.crop_size]


class ReshapeLayer(nn.Module):
    def __init__(self, shape):
        super(ReshapeLayer, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(768, 3)
        self.fc_log_var = nn.Linear(768, 3)
        self.sampling = SamplingLayer()

        self.reshape = ReshapeLayer((-1, 64, 12)),
        self.cropping1d_41 = Cropping1D(23),

        # 解码器部分
        self.decoder_input = nn.Linear(3, 768)
        self.decoder = nn.Sequential(
            ReshapeLayer((-1, 64, 12)),
            nn.ConvTranspose1d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=1, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=2, stride=1, padding=0),
            Cropping1D(23),
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var

    def decode(self, z):
        z = self.decoder_input(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.sampling(mu, log_var)
        return self.decode(z), mu, log_var


if "__main__" == __name__:
    # 模型实例化和测试输入
    model = VAE()
    print(model)

    # 假设有一个输入
    input_tensor = torch.randn(2, 1, 23)  # [batch_size, channels, length]
    output, mu, log_var = model(input_tensor)
    print(output.shape)  # 应该与输入维度相符
