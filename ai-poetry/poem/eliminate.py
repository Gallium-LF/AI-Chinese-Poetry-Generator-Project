"""此代码用于出重"""
import os

# 获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 指定源文件和目标文件的路径
source_file = os.path.join(current_dir, 'dataSet.txt')
target_file = os.path.join(current_dir, 'output.txt')

# 打开源文件进行读取
with open(source_file, 'r', encoding='utf-8') as fi:
    txt = fi.readlines()

source_file = 'dataSet.txt'
target_file = 'output.txt'

# 打开源文件进行读取
with open(source_file, 'r', encoding='utf-8') as fi:
    txt = fi.readlines()

# 遍历源文件中的每一行，将非重复行写入目标文件
with open(target_file, 'w', encoding='utf-8') as fo:
    for w in txt:
        with open(target_file, 'r', encoding='utf-8') as f:
            txt2 = f.readlines()
            if w not in txt2:
                fo.write(w)
            else:
                print("已去除重复 --> " + w)