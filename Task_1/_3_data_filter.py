import codecs
import csv
import math
import re

from db import get_list_by_status
from nltk.corpus import stopwords

punctuation = r"""!"#$%&'()*+,？-./:;<=>?@[\]^_`{|}~！"""
stopwords_english = stopwords.words('english')


def filter_result(size=10000, skip=0):
    data = get_list_by_status(1, size, skip)
    data_valid_data = []

    for d in data:
        # 把停用词过滤掉

        title = d['title'].translate(str.maketrans('', '', punctuation))  # 去除标点符号
        # title = title.replace(' ', '')  # 中间有空格
        comment = d['comment'].translate(str.maketrans('', '', punctuation))  # 去除标点符号
        comment = re.sub(r"(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b", '', comment, flags=re.MULTILINE)
        filtered = [w for w in comment.split(" ") if (w not in stopwords_english)]
        label = d['url'].split("/")[0]
        if len(comment) > 2000:
            data_valid_data.append({
                "label": label,
                "title": title,
                "comment": " ".join(filtered),
            })
    save_test_result(data_valid_data)


def save_test_result(data_list):
    """
    保存预测结果
    """
    csv_file = codecs.open('../data/analysis_data.csv', 'a+', encoding='utf-8')
    writer = csv.writer(csv_file)
    for index, d in enumerate(data_list):
        writer.writerow((d['label'], d['title'], len(d['title']), d['comment'], len(d['comment'])))
    csv_file.close()


def data_handle(size):
    page_size = 5000
    if size > page_size:
        count = math.ceil(size / page_size)
        for i in range(count):
            print(i, i * page_size)
            filter_result((i + 1) * page_size, i * page_size)


if __name__ == '__main__':
    # # 过滤 特殊符号 并且保存
    data_handle(30000)
    # import nltk
    # nltk.download() #
