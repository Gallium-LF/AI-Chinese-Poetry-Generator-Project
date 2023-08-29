"""此代码用于测试生成诗"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 加载词汇表和映射
with open('dataSet.txt', 'r', encoding='utf-8') as file:
    poems = file.read().splitlines()

vocab = sorted(set(''.join(poems)))
char2idx = {char: idx for idx, char in enumerate(vocab)}
idx2char = {idx: char for idx, char in enumerate(vocab)}

# 载入模型
def GRU_model(vocab_size, embedding_dim, units, batch_size):
    model = keras.Sequential([
        layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        layers.GRU(units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
        layers.Dense(vocab_size)
    ])
    return model

vocab_size = len(idx2char)
embedding_dim = 64
units = 128
BATCH_SIZE = 1  # 一次生成一首诗

model = GRU_model(vocab_size, embedding_dim, units, BATCH_SIZE)
checkpoint_dir = './training_checkpoints'

# 初始化模型权重
model.build(tf.TensorShape([BATCH_SIZE, None]))
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

# 生成一首诗
def generate_poem(model, start_string, temperature=1.0, max_length=100):
    input_eval = [char2idx[char] for char in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    generated_text = []

    model.reset_states()
    for _ in range(max_length):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        generated_text.append(idx2char[predicted_id])

    return start_string + ''.join(generated_text)

# 根据参数生成诗
def generate_poem_choice(state, start):
    poem = ''
    if state == '0':
        for _ in range(len(start)):
            if _ % 2 == 0:
                poem += generate_poem(model, start_string=start[_], temperature=1, max_length=4) + '，'
            else:
                if _ == 3:
                    poem += generate_poem(model, start_string=start[_], temperature=1, max_length=4) + '。\n'
                else:
                    poem += generate_poem(model, start_string=start[_], temperature=1, max_length=4) + '。'
    elif state == '1':
        poem = generate_poem(model, start_string=start, temperature=1, max_length=20 - len(start))
        poem = poem[:5] + '，' + poem[5:10] + '。' + poem[10:15] + '，' + poem[15:20] + '。' + poem[20:]

    return poem

def parse_input(user_input):
    if '律诗' in user_input:
        return '律诗'
    elif '绝句' in user_input:
        return '绝句'
    else:
        return None

print("我是一个五言诗人，请让我为你写诗。。。")

while True:
    user_input = input("请告诉我你想要的诗的类型，比如：绝句或律诗，或输入'退出'以结束程序：\n")
    if user_input == '退出':
        break

    poem_type = parse_input(user_input)
    if poem_type is None:
        print("抱歉，我没有理解你的要求，请尝试输入'律诗'或'绝句'")
        continue

    state = input("请问想要我为你写一首藏头诗还是续写诗？\n")
    if '藏头诗' in state:
        while True:
            start_string = input("请输入关键词:\n")
            if poem_type == '律诗':
                # 生成八句话的藏头诗
                if len(start_string) != 8:
                    print('律诗的藏头诗要藏八个字哦~')
                    continue
                else:
                    poem = generate_poem_choice('0', start_string[:8])
                    break
            elif poem_type == '绝句':
                # 生成四句话的藏头诗
                if len(start_string) != 4:
                    print('绝句的藏头诗要藏四个字哦~')
                    continue
                else:
                    poem = generate_poem_choice('0', start_string[:4])
                    break
    elif '续写诗' in state:
        start_string = input("请输入开始字符串:\n")
        if poem_type == '律诗':
            # 生成八句话的续写诗
            poem = generate_poem(model, start_string, max_length=40-len(start_string))
            poem = poem[:5] + '，' + poem[5:10] + '。' + poem[10:15] + '，' + poem[15:20] + '。\n' + poem[20:25] + '，' +poem[25:30] + '。'+ poem[30:35] + '，' + poem[35:40] + '。' + poem[40:]
        elif poem_type == '绝句':
            # 生成四句话的续写诗
            poem = generate_poem(model, start_string, max_length=20-len(start_string))
            poem = poem[:5] + '，' + poem[5:10] + '。' + poem[10:15] + '，' + poem[15:20] + '。' + poem[20:]
    else:
        print("抱歉，我没有理解你的要求，请尝试输入'藏头诗'、'续写诗'或'主题'")
        continue

    print('以下是我为你写的诗：')
    print(poem)