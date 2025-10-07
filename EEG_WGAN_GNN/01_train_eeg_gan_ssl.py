import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. 配置 ---
config = {
    "csv_path": "your_eeg_data.csv",
    "channels": ["Channel1", "Channel2", "Channel3", "Channel4"],
    "window_size": 256,
    "latent_dim": 100,
    "batch_size": 64,
    "epochs": 15000,
    "lr": 0.00015,
    "b1": 0.5,
    "b2": 0.9,
    "n_critic": 5,
    "gp_lambda": 10,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}
print(f"Using device: {config['device']}")

# --- 2. 模型定义 ---
class Generator(nn.Module):
    def __init__(self, latent_dim, channels, size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # 输入: (B, latent_dim, 1) -> 输出: (B, 256, 16)
            nn.ConvTranspose1d(latent_dim, 256, 16, 1, 0), nn.BatchNorm1d(256), nn.ReLU(True),
            # 输出: (B, 128, 32)
            nn.ConvTranspose1d(256, 128, 4, 2, 1), nn.BatchNorm1d(128), nn.ReLU(True),
            # 输出: (B, 64, 64)
            nn.ConvTranspose1d(128, 64, 4, 2, 1), nn.BatchNorm1d(64), nn.ReLU(True),
            # 输出: (B, 32, 128)
            nn.ConvTranspose1d(64, 32, 4, 2, 1), nn.BatchNorm1d(32), nn.ReLU(True),
            # 输出: (B, channels, 256)
            nn.ConvTranspose1d(32, channels, 4, 2, 1), nn.Tanh()
        )
    def forward(self, z):
        return self.model(z.view(z.size(0), -1, 1))

class Critic(nn.Module):
    def __init__(self, channels, size):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            # 输入: (B, channels, 256) -> 输出: (B, 32, 128)
            nn.Conv1d(channels, 32, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            # 输出: (B, 64, 64)
            nn.Conv1d(32, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            # 输出: (B, 128, 32)
            nn.Conv1d(64, 128, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            # 输出: (B, 256, 16)
            nn.Conv1d(128, 256, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            # 展平后输出一个分数
            nn.Flatten(),
            nn.Linear(256 * 16, 1)
        )
    def forward(self, x):
        return self.model(x)

# --- 3. 工具函数 ---
def load_data(path, channels, window_size, batch_size):
    """加载并预处理数据"""
    if not os.path.exists(path):
        print(f"找不到文件 '{path}', 创建一个随机演示文件。")
        dummy_df = pd.DataFrame(np.random.randn(5000, len(channels)), columns=channels)
        dummy_df.to_csv(path, index=False)
    
    df = pd.read_csv(path)
    data = df[channels].values
    
    # 归一化到 [-1, 1]
    min_val, max_val = data.min(), data.max()
    data = 2 * (data - min_val) / (max_val - min_val) - 1
    
    # 窗口化
    segments = np.array([data[i:i+window_size] for i in range(len(data) - window_size)])
    # 转换形状为 (N, C, L)
    data_tensor = torch.tensor(segments, dtype=torch.float32).permute(0, 2, 1)
    
    dataloader = DataLoader(data_tensor, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader, min_val, max_val

def compute_gradient_penalty(critic, real, fake, device):
    """计算梯度惩罚"""
    alpha = torch.rand(real.size(0), 1, 1, device=device).expand_as(real)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    prob_interpolated = critic(interpolated)
    
    gradients = torch.autograd.grad(
        outputs=prob_interpolated, inputs=interpolated,
        grad_outputs=torch.ones_like(prob_interpolated),
        create_graph=True, retain_graph=True
    )[0].view(real.size(0), -1)
    
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

def generate_and_plot(generator, epoch, min_val, max_val, config):
    """生成样本并绘图保存"""
    generator.eval()
    with torch.no_grad():
        z = torch.randn(4, config["latent_dim"], device=config["device"])
        generated_eeg = generator(z).cpu().numpy()
    generator.train()

    # 反归一化
    generated_eeg = (generated_eeg + 1) * (max_val - min_val) / 2 + min_val
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f'Generated EEG Samples at Epoch {epoch}', fontsize=16)
    time_axis = np.arange(config["window_size"])
    
    for i in range(4):
        for j in range(len(config["channels"])):
            axes[i].plot(time_axis, generated_eeg[i, j, :], label=f'Ch {j+1}')
        axes[i].legend(loc='upper right')
    
    plt.xlabel('Time Points')
    os.makedirs("generated_samples", exist_ok=True)
    plt.savefig(f"generated_samples/epoch_{epoch}.png")
    plt.close()

# --- 4. 训练流程 ---
def main():
    # 准备数据和模型
    dataloader, min_val, max_val = load_data(
        config["csv_path"], config["channels"], config["window_size"], config["batch_size"]
    )
    generator = Generator(config["latent_dim"], len(config["channels"]), config["window_size"]).to(config["device"])
    critic = Critic(len(config["channels"]), config["window_size"]).to(config["device"])

    opt_g = optim.Adam(generator.parameters(), lr=config["lr"], betas=(config["b1"], config["b2"]))
    opt_c = optim.Adam(critic.parameters(), lr=config["lr"], betas=(config["b1"], config["b2"]))

    print("Starting training...")
    for epoch in range(1, config["epochs"] + 1):
        for i, real_eegs in enumerate(dataloader):
            real_eegs = real_eegs.to(config["device"])
            
            # --- 训练评论家 ---
            opt_c.zero_grad()
            z = torch.randn(real_eegs.size(0), config["latent_dim"], device=config["device"])
            fake_eegs = generator(z).detach()
            
            gp = compute_gradient_penalty(critic, real_eegs, fake_eegs, config["device"])
            loss_c = critic(fake_eegs).mean() - critic(real_eegs).mean() + config["gp_lambda"] * gp
            
            loss_c.backward()
            opt_c.step()

            # --- 训练生成器 ---
            if i % config["n_critic"] == 0:
                opt_g.zero_grad()
                z_gen = torch.randn(real_eegs.size(0), config["latent_dim"], device=config["device"])
                loss_g = -critic(generator(z_gen)).mean()
                loss_g.backward()
                opt_g.step()

        # --- 定期输出和保存 ---
        if epoch % 100 == 0:
            print(f"[Epoch {epoch}/{config['epochs']}] [C loss: {loss_c.item():.4f}] [G loss: {loss_g.item():.4f}]")
            generate_and_plot(generator, epoch, min_val, max_val, config)
            torch.save(generator.state_dict(), "generator.pth")

if __name__ == '__main__':
    main()