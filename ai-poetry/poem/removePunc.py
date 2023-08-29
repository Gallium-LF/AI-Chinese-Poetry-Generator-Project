"""此代码用于去除收集到的诗中的标点并将每一句放在一行"""
import string
import re

def remove_punctuation(text):
    # 去除标点符号和空格（包括中文标点和空格）
    punctuation_pattern = r'[{}]+'.format(re.escape(string.punctuation + '。，“”‘’！？【】（）《》 \n'))
    text = re.sub(punctuation_pattern, '', text)
    return text

def split_into_lines(text, line_length):
    # 将字符串按照每行指定长度进行分割
    lines = [text[i:i+line_length] for i in range(0, len(text), line_length)]
    return lines

file_path = 'wuyan.txt'  # 输入文件路径
output_path = 'new.txt'  # 输出文件路径
line_length = 5  # 每行的字符长度

with open(file_path, 'r',encoding='utf-8') as file:
    content = file.read()
    content = remove_punctuation(content)  # 去除标点符号
    lines = split_into_lines(content, line_length)  # 分割成每行指定长度的字符串

with open(output_path, 'a',encoding='utf-8') as file:
    for line in lines:
        file.write(line + '\n')