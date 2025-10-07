# -*- coding: utf-8 -*-
"""
WGAN-GP for EEG Data Augmentation using TensorFlow Keras.

This script trains and saves a separate WGAN-GP model for each of the 4 EEG channels.
It explicitly writes out the training process for each channel without using a loop
for maximum clarity and allows for saving individual model weights.

Author: 黑心莲
Env：eeggangpu_old
Date: 2025-07-08
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import os
from sklearn.utils import shuffle

print(f"TensorFlow Version: {tf.__version__}")
print(f"Num GPUs Available: {len(tf.config.experimental.list_physical_devices('GPU'))}")
if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
    print("\nGPU is available and will be used for training.")
else:
    print("\nWarning: GPU not found. Training will run on CPU and may be slow.")


# ==============================================================================
# 1. 数据预处理 (Data Preprocessing) - [No changes]
# ==============================================================================

def preprocess_eeg_data(file_path, timesteps_per_sample=256):
    """
    Loads, cleans, and segments EEG data. Also returns channel labels.
    """
    if not os.path.exists(file_path):
        print(f"错误：找不到文件 '{file_path}'。")
        return None, None

    print(f"正在从 '{file_path}' 加载数据...")
    try:
        data = pd.read_csv(file_path, sep=',')
    except Exception as e:
        print(f"错误：无法使用逗号解析文件。请检查文件格式。错误: {e}")
        return None, None

    print("成功加载数据。检测到的列名:", data.columns.tolist())
    
    try:
        data = data.drop(columns=['Timestamp', 'PacketType', 'Data'])
    except KeyError:
        print("致命错误：无法删除列 'Timestamp', 'PacketType', 'Data'。请检查列名。")
        return None, None

    print(f"成功删除列后，剩余数据维度 (行数, 通道数): {data.shape}")
    num_channels = data.shape[1]

    num_samples_possible = len(data) // timesteps_per_sample
    print(f"可以切分出 {num_samples_possible} 个完整的1秒样本。")

    if num_samples_possible == 0:
        print("错误：数据量不足以切分出任何一个完整的样本。")
        return None, None
    
    num_rows_to_keep = num_samples_possible * timesteps_per_sample
    data_truncated = data.iloc[:num_rows_to_keep]
    data_reshaped = data_truncated.values.reshape(num_samples_possible, timesteps_per_sample, num_channels)
    data_swapped = data_reshaped.transpose(0, 2, 1)
    
    num_total_samples = data_swapped.shape[0] * data_swapped.shape[1]
    all_channels_as_samples = data_swapped.reshape(num_total_samples, timesteps_per_sample)
    processed_data = np.expand_dims(all_channels_as_samples, axis=-1)

    labels = np.array([i for i in range(num_channels) for _ in range(num_samples_possible)])

    print(f"\n预处理完成。最终数据 Shape: {processed_data.shape}, 标签 Shape: {labels.shape}")
    return processed_data.astype('float32'), labels.astype('int32')

# ==============================================================================
# 2. 模型构建 (WGAN-GP Model Building) - [No changes]
# ==============================================================================

# Hyperparameters
EEG_TIMESTEPS = 256
EEG_FEATURES = 1
INPUT_SHAPE = (EEG_TIMESTEPS, EEG_FEATURES)
BATCH_SIZE = 64
EPOCHS_PER_CHANNEL = 200 
LATENT_DIM = 100 
GP_WEIGHT = 10.0
LEARNING_RATE = 0.0002
BETA_1 = 0.5
BETA_2 = 0.9

def build_generator(latent_dim):
    model = keras.Sequential(name="Generator")
    model.add(layers.Dense(32 * 16, input_dim=latent_dim))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Reshape((32, 16)))
    model.add(layers.Conv1DTranspose(128, kernel_size=4, strides=2, padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv1DTranspose(128, kernel_size=4, strides=2, padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv1DTranspose(64, kernel_size=4, strides=2, padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv1D(EEG_FEATURES, kernel_size=7, padding="same", activation="tanh"))
    return model

def build_critic(input_shape):
    model = keras.Sequential(name="Critic")
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv1D(64, kernel_size=5, strides=2, padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv1D(128, kernel_size=5, strides=2, padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv1D(256, kernel_size=5, strides=2, padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1))
    return model

class WGAN_GP(keras.Model):
    def __init__(self, generator, critic, latent_dim, gp_weight):
        super().__init__()
        self.generator = generator
        self.critic = critic
        self.latent_dim = latent_dim
        self.gp_weight = gp_weight
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    def compile(self, g_optimizer, d_optimizer):
        super().compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def gradient_penalty(self, batch_size, real_samples, fake_samples):
        alpha = tf.random.normal([batch_size, 1, 1], 0.0, 1.0)
        interpolated = real_samples + alpha * (fake_samples - real_samples)
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.critic(interpolated, training=True)
        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real_samples):
        batch_size = tf.shape(real_samples)[0]
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

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            generated_samples = self.generator(random_latent_vectors, training=True)
            gen_output = self.critic(generated_samples, training=True)
            g_loss = -tf.reduce_mean(gen_output)
        g_grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))

        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {"d_loss": self.d_loss_metric.result(), "g_loss": self.g_loss_metric.result()}

# ==============================================================================
# 3. 训练与执行 (Training & Execution) - NEW SEPARATE STRUCTURE
# ==============================================================================
def main():
    # --- 1. 数据准备 ---
    eeg_file_path = 'data5min.csv'
    real_eeg_data, real_labels = preprocess_eeg_data(eeg_file_path, timesteps_per_sample=EEG_TIMESTEPS)
    
    if real_eeg_data is None:
        return
        
    max_abs_val = np.max(np.abs(real_eeg_data))
    if max_abs_val > 0:
        real_eeg_data = real_eeg_data / max_abs_val
    print(f"\n数据已归一化到 [-1, 1] 范围 (除以最大绝对值: {max_abs_val:.2f})")

    # 用于存储所有生成的假样本
    all_fake_samples = []

    # --------------------------------------------------------------------------
    #                       模型 1: 训练通道 0
    # --------------------------------------------------------------------------
    print(f"\n{'='*25} 开始处理通道 1 (标签 0) {'='*25}")
    x_train_channel_0 = real_eeg_data[real_labels == 0]
    train_dataset_0 = tf.data.Dataset.from_tensor_slices(x_train_channel_0).shuffle(len(x_train_channel_0)).batch(BATCH_SIZE)
    
    # 创建模型
    generator_0 = build_generator(LATENT_DIM)
    critic_0 = build_critic(INPUT_SHAPE)
    wgan_0 = WGAN_GP(generator_0, critic_0, LATENT_DIM, GP_WEIGHT)
    wgan_0.compile(
        g_optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2),
        d_optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2)
    )
    # 训练
    print(f"开始为通道 1 训练模型...")
    wgan_0.fit(train_dataset_0, epochs=EPOCHS_PER_CHANNEL, verbose=2)
    # 保存模型权重
    generator_0.save_weights('generator_channel_0_weights.h5')
    print("通道 1 的生成器权重已保存到 'generator_channel_0_weights.h5'")
    # 生成假样本
    num_to_generate_0 = len(x_train_channel_0)
    fake_samples_0 = generator_0.predict(tf.random.normal(shape=(num_to_generate_0, LATENT_DIM)))
    all_fake_samples.append(fake_samples_0)

    # --------------------------------------------------------------------------
    #                       模型 2: 训练通道 1
    # --------------------------------------------------------------------------
    print(f"\n{'='*25} 开始处理通道 2 (标签 1) {'='*25}")
    x_train_channel_1 = real_eeg_data[real_labels == 1]
    train_dataset_1 = tf.data.Dataset.from_tensor_slices(x_train_channel_1).shuffle(len(x_train_channel_1)).batch(BATCH_SIZE)

    # 创建模型
    generator_1 = build_generator(LATENT_DIM)
    critic_1 = build_critic(INPUT_SHAPE)
    wgan_1 = WGAN_GP(generator_1, critic_1, LATENT_DIM, GP_WEIGHT)
    wgan_1.compile(
        g_optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2),
        d_optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2)
    )
    # 训练
    print(f"开始为通道 2 训练模型...")
    wgan_1.fit(train_dataset_1, epochs=EPOCHS_PER_CHANNEL, verbose=2)
    # 保存模型权重
    generator_1.save_weights('generator_channel_1_weights.h5')
    print("通道 2 的生成器权重已保存到 'generator_channel_1_weights.h5'")
    # 生成假样本
    num_to_generate_1 = len(x_train_channel_1)
    fake_samples_1 = generator_1.predict(tf.random.normal(shape=(num_to_generate_1, LATENT_DIM)))
    all_fake_samples.append(fake_samples_1)

    # --------------------------------------------------------------------------
    #                       模型 3: 训练通道 2
    # --------------------------------------------------------------------------
    print(f"\n{'='*25} 开始处理通道 3 (标签 2) {'='*25}")
    x_train_channel_2 = real_eeg_data[real_labels == 2]
    train_dataset_2 = tf.data.Dataset.from_tensor_slices(x_train_channel_2).shuffle(len(x_train_channel_2)).batch(BATCH_SIZE)
    
    # 创建模型
    generator_2 = build_generator(LATENT_DIM)
    critic_2 = build_critic(INPUT_SHAPE)
    wgan_2 = WGAN_GP(generator_2, critic_2, LATENT_DIM, GP_WEIGHT)
    wgan_2.compile(
        g_optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2),
        d_optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2)
    )
    # 训练
    print(f"开始为通道 3 训练模型...")
    wgan_2.fit(train_dataset_2, epochs=EPOCHS_PER_CHANNEL, verbose=2)
    # 保存模型权重
    generator_2.save_weights('generator_channel_2_weights.h5')
    print("通道 3 的生成器权重已保存到 'generator_channel_2_weights.h5'")
    # 生成假样本
    num_to_generate_2 = len(x_train_channel_2)
    fake_samples_2 = generator_2.predict(tf.random.normal(shape=(num_to_generate_2, LATENT_DIM)))
    all_fake_samples.append(fake_samples_2)

    # --------------------------------------------------------------------------
    #                       模型 4: 训练通道 3
    # --------------------------------------------------------------------------
    print(f"\n{'='*25} 开始处理通道 4 (标签 3) {'='*25}")
    x_train_channel_3 = real_eeg_data[real_labels == 3]
    train_dataset_3 = tf.data.Dataset.from_tensor_slices(x_train_channel_3).shuffle(len(x_train_channel_3)).batch(BATCH_SIZE)

    # 创建模型
    generator_3 = build_generator(LATENT_DIM)
    critic_3 = build_critic(INPUT_SHAPE)
    wgan_3 = WGAN_GP(generator_3, critic_3, LATENT_DIM, GP_WEIGHT)
    wgan_3.compile(
        g_optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2),
        d_optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2)
    )
    # 训练
    print(f"开始为通道 4 训练模型...")
    wgan_3.fit(train_dataset_3, epochs=EPOCHS_PER_CHANNEL, verbose=2)
    # 保存模型权重
    generator_3.save_weights('generator_channel_3_weights.h5')
    print("通道 4 的生成器权重已保存到 'generator_channel_3_weights.h5'")
    # 生成假样本
    num_to_generate_3 = len(x_train_channel_3)
    fake_samples_3 = generator_3.predict(tf.random.normal(shape=(num_to_generate_3, LATENT_DIM)))
    all_fake_samples.append(fake_samples_3)

    # --------------------------------------------------------------------------
    #                   最终数据增强与保存
    # --------------------------------------------------------------------------
    print(f"\n{'='*25} 数据增强与保存 {'='*25}")
    
    fake_data_all = np.concatenate(all_fake_samples, axis=0)
    fake_labels = real_labels
    
    if max_abs_val > 0:
        real_eeg_data_rescaled = real_eeg_data * max_abs_val
        fake_data_rescaled = fake_data_all * max_abs_val
    else:
        real_eeg_data_rescaled = real_eeg_data
        fake_data_rescaled = fake_data_all

    augmented_data = np.concatenate([real_eeg_data_rescaled, fake_data_rescaled], axis=0)
    augmented_labels = np.concatenate([real_labels, fake_labels], axis=0)

    augmented_data, augmented_labels = shuffle(augmented_data, augmented_labels, random_state=42)
    print(f"数据增强完成。总样本数: {len(augmented_data)}")

    output_filename = 'augmented_eeg_data.csv'
    print(f"正在将增强后的数据保存到 '{output_filename}'...")
    
    data_to_save = np.squeeze(augmented_data, axis=-1)
    df_to_save = pd.DataFrame(data_to_save)
    df_to_save.insert(0, 'Channel_Label', augmented_labels)
    
    df_to_save.to_csv(output_filename, index=False)
    print(f"保存成功！文件 '{output_filename}' 已创建。")

if __name__ == '__main__':
    main()