"""此代码用于检测模型的拟合效果"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from tensorflow.keras import regularizers

# 读取数据集
with open('dataSet.txt', 'r', encoding='utf-8') as file:
    poems = file.read().splitlines()

# 构建词汇表
vocab = sorted(set(''.join(poems)))

# 创建词汇表和索引的映射
char2idx = {char: idx for idx, char in enumerate(vocab)}
idx2char = {idx: char for idx, char in enumerate(vocab)}

# 将诗句转换为索引序列
data = []
for poem in poems:
    data.append([char2idx[char] for char in poem])

data_line = np.array([word for poem in data for word in poem])

# 构建模型的函数
def GRU_model(vocab_size, embedding_dim, units, batch_size, l2_reg=0.001):
    model = keras.Sequential([
        layers.Embedding(vocab_size,
                         embedding_dim,
                         batch_input_shape=[batch_size, None]),
        layers.GRU(units,
                   return_sequences=True,
                   stateful=True,
                   recurrent_initializer='glorot_uniform',
                   kernel_regularizer=regularizers.l2(l2_reg)),
        layers.Dense(vocab_size)
    ])
    return model

# 切分成输入和输出
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

# 损失函数
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels,
                                                           logits,
                                                           from_logits=True)


# 每批大小
BATCH_SIZE = 16
# 缓冲区大小
BUFFER_SIZE = 10000
# 训练周期
EPOCHS = 50
# 诗的长度
poem_size = 5
# 嵌入的维度
embedding_dim = 64
# RNN 的单元数量
units = 128
# 正则化强度
L2_REG = 0.001


# 划分训练集和验证集
train_size = int(0.95 * len(data_line))
train_data = data_line[:train_size]
val_data = data_line[train_size:]

# 创建训练集和验证集的数据集
train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
train_poems = train_dataset.batch(poem_size + 1, drop_remainder=True)
train_dataset = train_poems.map(split_input_target)
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

val_dataset = tf.data.Dataset.from_tensor_slices(val_data)
val_poems = val_dataset.batch(poem_size + 1, drop_remainder=True)
val_dataset = val_poems.map(split_input_target)
val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)

# 创建模型
model = GRU_model(vocab_size=len(idx2char),
                  embedding_dim=embedding_dim,
                  units=units,
                  batch_size=BATCH_SIZE,
                  l2_reg=L2_REG)
model.summary()
model.compile(optimizer='adam', loss=loss)

# 检查点目录
checkpoint_dir = './training_checkpoints'
# 检查点设置
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix, save_weights_only=True)

# 记录训练集和验证集的损失函数值
history = model.fit(train_dataset, epochs=EPOCHS, 
                    validation_data=val_dataset, 
                    callbacks=[checkpoint_callback])

# 绘制学习曲线
import matplotlib.pyplot as plt

train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_loss) + 1)

plt.plot(epochs, train_loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()