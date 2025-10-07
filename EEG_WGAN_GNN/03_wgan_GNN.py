# -*- coding: utf-8 -*-
"""
WGAN-GP for 4-Channel EEG Data Augmentation
This script trains a WGAN-GP to generate synthetic 4-channel EEG data.
"""

# ==============================================================================
# 故事的开篇：召唤所有需要的“神殿”与“卷轴”
# ==============================================================================
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

# ==============================================================================
# 第一章：神匠的蓝图 (全局设定与超参数)
# ==============================================================================
# --- 数据与形状设定 (可根据你的需求修改) ---
EEG_CHANNELS = 4      # 我们的EEG设备有4个通道
EEG_TIMESTEPS = 256   # 每个样本的时间步长 (1秒 @ 256Hz)
INPUT_SHAPE = (EEG_TIMESTEPS, EEG_CHANNELS) # 评判者(Critic)的输入形状

# --- 训练过程设定 (可根据你的需求修改) ---
BATCH_SIZE = 32         # 魔法熔炉的大小：每次处理32个样本
EPOCHS = 50             # 完整试炼的纪元数 (先用一个较小的值来跑通验证)
LATENT_DIM = 128        # 伪造者(Generator)的灵感之源“混沌空间”的维度

# --- WGAN-GP 核心参数 ---
GP_WEIGHT = 10.0        # “秩序神罚”（梯度惩罚）的权重

# --- 优化器（教练）参数 ---
LEARNING_RATE = 0.0002
BETA_1 = 0.5
BETA_2 = 0.9

# ==============================================================================
# 准备工作：确保有数据可用 (如果文件不存在，则创建一个虚拟文件)
# ==============================================================================
def create_dummy_data_if_not_exists(filename="data5min.csv"):
    if not os.path.exists(filename):
        print(f"警告: 未找到 '{filename}'。正在创建一个虚拟数据文件用于测试...")
        # 创建一个足够长的虚拟数据集以供切分
        num_rows = 2000 
        timestamps = np.linspace(1.75138E+15, 1.75138E+15 + num_rows, num_rows)
        data = {
            'Timestamp': timestamps,
            'PacketType': ['EEG'] * num_rows,
            'Data': ['dummy_data'] * num_rows,
            # 使用正弦波和噪声模拟真实的EEG数据
            'Channel1': 500 + 100 * np.sin(np.arange(num_rows) * 0.1) + np.random.randn(num_rows) * 20,
            'Channel2': 700 + 80 * np.sin(np.arange(num_rows) * 0.15) + np.random.randn(num_rows) * 25,
            'Channel3': 700 - 90 * np.cos(np.arange(num_rows) * 0.1) + np.random.randn(num_rows) * 22,
            'Channel4': 600 + 110 * np.cos(np.arange(num_rows) * 0.12) + np.random.randn(num_rows) * 28,
        }
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"虚拟文件 '{filename}' 已创建。")

# ==============================================================================
# 第二章：预处理仪式 (数据清洗、切分与标准化)
# ==============================================================================
def preprocess_eeg_data(file_path, timesteps=256, num_channels=4):
    print(f"\n{'='*20} 开始预处理仪式 {'='*20}")
    if not os.path.exists(file_path):
        print(f"错误：找不到文件 '{file_path}'。")
        return None

    # --- 1. 数据加载与清洗 ---
    print(f"正在从 '{file_path}' 加载数据...")
    df = pd.read_csv(file_path)
    
    # 根据你的要求，我们只保留4个通道的数据
    # Timestamp在固定采样率下是冗余信息，可以舍弃
    channel_columns = [f'Channel{i+1}' for i in range(num_channels)]
    if not all(col in df.columns for col in channel_columns):
        print(f"错误：CSV文件中缺少必要的通道列。需要: {channel_columns}")
        return None
        
    eeg_data = df[channel_columns].values
    print("数据清洗完成，只保留4个EEG通道。")

    # --- 2. 数据切分 (窗口化) ---
    print(f"正在将数据切分为长度为 {timesteps} 的样本...")
    num_samples = len(eeg_data) - timesteps + 1
    samples = []
    for i in range(num_samples):
        sample = eeg_data[i : i + timesteps]
        samples.append(sample)
    
    # 将样本列表转换为一个巨大的Numpy数组
    # 最终形状: (样本总数 N, 时间步长 256, 通道数 4)
    real_samples_np = np.array(samples)
    print(f"数据切分完成，共得到 {real_samples_np.shape[0]} 个样本。")

    # --- 3. 标准化 ("失落的序章" - 将数据缩放到 [-1, 1]) ---
    print("正在进行标准化（归一化到 [-1, 1] 区间）...")
    # MinMaxScaler需要2D数据，所以我们先临时重塑数据
    original_shape = real_samples_np.shape
    real_samples_reshaped = real_samples_np.reshape(-1, num_channels)
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(real_samples_reshaped) # 学习所有数据的最大最小值
    normalized_data_reshaped = scaler.transform(real_samples_reshaped)
    
    # 将数据恢复到原始的3D形状
    normalized_data = normalized_data_reshaped.reshape(original_shape)
    
    print(f"标准化完成。最终数据形状: {normalized_data.shape}")
    print(f"{'='*20} 预处理仪式结束 {'='*20}\n")
    
    # 返回处理好的数据和用于未来还原的scaler对象
    return normalized_data, scaler

# ==============================================================================
# 第三章：铸造神兵（模型构建函数）
# ==============================================================================

# --- 3.1 “伪造者”生成器 ---
def build_generator(latent_dim, num_channels):
    model = keras.Sequential(name="Generator")
    model.add(layers.Dense(32 * 16, input_dim=latent_dim))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Reshape((16, 32))) # 初始形态 (长度, 深度)
    
    # 逆向卷积，从小变大，从抽象到具体
    # 16x32 -> 32x64
    model.add(layers.Conv1DTranspose(64, kernel_size=4, strides=2, padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    # 32x64 -> 64x128
    model.add(layers.Conv1DTranspose(128, kernel_size=4, strides=2, padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    # 64x128 -> 128x128
    model.add(layers.Conv1DTranspose(128, kernel_size=4, strides=2, padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    # 128x128 -> 256x64
    model.add(layers.Conv1DTranspose(64, kernel_size=4, strides=2, padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    
    # 最终打磨层，输出我们想要的通道数和数据范围
    model.add(layers.Conv1D(num_channels, kernel_size=7, padding="same", activation="tanh"))
    
    # 确保输出形状是我们期望的 (256, 4)
    assert model.output_shape == (None, EEG_TIMESTEPS, num_channels)
    return model

# --- 3.2 “评判者” ---
def build_critic(input_shape):
    model = keras.Sequential(name="Critic")
    model.add(layers.Input(shape=input_shape))
    
    # 卷积层，从大变小，从具体到抽象
    # 256x4 -> 128x64
    model.add(layers.Conv1D(64, kernel_size=5, strides=2, padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    # 128x64 -> 64x128
    model.add(layers.Conv1D(128, kernel_size=5, strides=2, padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    # 64x128 -> 32x128
    model.add(layers.Conv1D(128, kernel_size=5, strides=2, padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    
    # 展平，为最终裁决做准备
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.3))
    
    # 最终裁决，输出一个无限制的分数
    model.add(layers.Dense(1))
    
    return model

# ==============================================================================
# 第四章：搭建宏伟的决斗场 (WGAN_GP核心类)
# ==============================================================================
class WGAN_GP(keras.Model):
    def __init__(self, generator, critic, latent_dim, gp_weight):
        super().__init__()
        self.generator = generator
        self.critic = critic
        self.latent_dim = latent_dim
        self.gp_weight = gp_weight
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def compile(self, g_optimizer, d_optimizer):
        super().compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

    def gradient_penalty(self, batch_size, real_samples, fake_samples):
        alpha = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0)
        interpolated = (alpha * real_samples) + ((1 - alpha) * fake_samples)
        
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.critic(interpolated, training=True)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real_samples):
        batch_size = tf.shape(real_samples)[0]

        # ------------------- 训练评判者 -------------------
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        
        with tf.GradientTape() as tape:
            fake_samples = self.generator(random_latent_vectors, training=True)
            real_output = self.critic(real_samples, training=True)
            fake_output = self.critic(fake_samples, training=True)
            
            d_cost = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
            gp = self.gradient_penalty(batch_size, real_samples, fake_samples)
            d_loss = d_cost + gp * self.gp_weight
            
        d_grads = tape.gradient(d_loss, self.critic.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_grads, self.critic.trainable_variables))

        # ------------------- 训练伪造者 -------------------
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        
        with tf.GradientTape() as tape:
            generated_samples = self.generator(random_latent_vectors, training=True)
            gen_output = self.critic(generated_samples, training=True)
            g_loss = -tf.reduce_mean(gen_output)
        
        g_grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))
        
        # 更新并返回记分牌上的平均分
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {"d_loss": self.d_loss_metric.result(), "g_loss": self.g_loss_metric.result()}

# ==============================================================================
# 最终章：主线任务 - 启动整个传奇
# ==============================================================================
def main():
    # --- 0. 检查环境与准备工作 ---
    print(f"Tensorflow Version: {tf.__version__}")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"\n发现 {len(gpus)} 个GPU, 将被用于训练。")
        except RuntimeError as e:
            print(e)
    else:
        print("\n警告: 未发现GPU。训练将在CPU上进行，可能会很慢。")
    
    create_dummy_data_if_not_exists()

    # --- 1. 数据准备 ---
    # 调用预处理仪式，得到可用于训练的“圣物”
    real_eeg_data, _ = preprocess_eeg_data(
        'data5min.csv', 
        timesteps=EEG_TIMESTEPS, 
        num_channels=EEG_CHANNELS
    )
    
    if real_eeg_data is None:
        print("数据预处理失败，程序中止。")
        return
        
    # 将Numpy数据转换为TensorFlow的Dataset对象，以获得最佳性能
    train_dataset = tf.data.Dataset.from_tensor_slices(real_eeg_data)
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    # --- 2. 模型构建与配置 ---
    print("\n--- 正在为多通道任务构建专属的决斗场 ---")
    # 召唤一位能创作4通道作品的“伪造者”
    generator = build_generator(LATENT_DIM, EEG_CHANNELS)
    # 召唤一位能鉴定4通道作品的“评判者”
    critic = build_critic(INPUT_SHAPE)
    
    # 将他们安置进宏伟的决斗场
    wgan = WGAN_GP(generator, critic, LATENT_DIM, GP_WEIGHT)
    
    # 为他们指派各自的“Adam”教练
    wgan.compile(
        g_optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2),
        d_optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2)
    )
    
    print("\n模型架构概览:")
    generator.summary()
    critic.summary()
    
    # --- 3. 开始训练！ ---
    print(f"\n--- 史诗对决开始！总计 {EPOCHS} 个纪元 ---")
    wgan.fit(train_dataset, epochs=EPOCHS)
    print("\n--- 对决结束，模仿大师已诞生！ ---")

    # --- 4. 保存与展示成果 ---
    # 保存训练好的“伪造者”的“灵魂”（权重）
    output_weights_file = 'generator_4channel_weights.h5'
    generator.save_weights(output_weights_file)
    print(f"\n生成器权重已保存到 '{output_weights_file}'")
    
    # 让新晋大师一展身手
    print("\n正在使用训练好的生成器创造10个新的伪造样本...")
    noise = tf.random.normal(shape=(10, LATENT_DIM))
    fake_samples = generator.predict(noise)
    print(f"成功生成了10个伪造样本，它们的形状是: {fake_samples.shape}")
    print("你可以加载权重，随时随地进行伪造。")
    print("\n整个传奇故事，圆满结束。")


if __name__ == '__main__':
    main()