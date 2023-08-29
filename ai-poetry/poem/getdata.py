"""此代码用于爬数据"""
import requests
from bs4 import BeautifulSoup

url_path = 'href.txt'

with open(url_path, 'r', encoding='utf-8') as file:
    url_list = file.readlines()

for url in url_list:
    # 目标网站的URL
    url = 'https://so.gushiwen.cn' + url
    url = url.replace('\n','')

    # 发起HTTP GET请求
    response = requests.get(url)

    # 解析HTML内容
    soup = BeautifulSoup(response.text, 'html.parser')

    # 找到包含古诗的元素，具体的CSS选择器根据目标网站的结构而定
    poem_elements = soup.select('.contson')

    # 保存古诗的文件路径
    file_path = 'wuyan.txt'

    # 遍历每个古诗元素并提取文本内容，并保存到文件中
    with open(file_path, 'a', encoding='utf-8') as file:
        for poem_element in poem_elements:
            poem_text = poem_element.text.strip()
            file.write(poem_text + '\n')