# For prerequisites running the following sample, visit https://help.aliyun.com/document_detail/611472.html
import re
from http import HTTPStatus
import dashscope
import os
from time import sleep
import json
from tqdm import tqdm
from multiprocessing import Pool
import traceback

os.chdir(os.path.dirname(os.path.abspath(__file__)))

dashscope.api_key = ""

def geninput(example):
    question = example['question']
    options = example['options']
    input = '请基于以下## 题目 内容，对题目进行知识点的分类:\n' \
            '' + f'## 题目{question}\n{options}\n' + '## 注意1. 知识点的类别只能是固定的【解析几何】【算术】【组合几何学】' \
                                                     '【组合数学】【计数】【画法几何学】【图论】【逻辑题】【度量几何角】【度量几何面积】' \
                                                     '【度量几何长度】【立体几何学】【统计数学】【拓扑学】【变换几何】【代数】十六个种类中的一类，' \
                                                     '**绝对不能有其他额外的类别\n**2. 请以## 本体知识点 开头完成分类，除此以外不要生成其他任何额外内容\n' \
                                                     '3. 只是进行知识点分类，**不需要对题目进行解答**4.输出格式是"本题知识点为："'
    return input


def load_jsonl(file_path) -> list:
    with open(file_path, 'r',encoding='UTF-8') as f:
        return [json.loads(line.strip()) for line in f]


def ask_qwvl(question):
    content = []
    content.append({"text": question})

    messages = [
        {
            "role"   : "user",
            "content": content
        }
    ]
    print(messages)
    response = dashscope.MultiModalConversation.call(model='qwen-vl-max',
                                                     messages=messages)

    if response.status_code == HTTPStatus.OK:
        pass
    else:

        print(response.code)  # The error code.
        print(response.message)  # The error message.

    response["input"] = messages
    return response


# Define the function to process each example
def process_example_merged_img(example):
    input = geninput(example)
    answer = ask_qwvl(input)
    answer['extra'] = example
    return answer


def test(path, output, processor, reset=False):
    if reset or not os.path.exists(output):
        with open(output, 'w',encoding='UTF-8') as f:
            pass
    with open(output, 'r',encoding='UTF-8') as f:
        processed_num = len(f.readlines())
    data = load_jsonl(path)[processed_num:]
    for example in tqdm(data):
        answer = processor(example)
        while answer["status_code"] != HTTPStatus.OK:
            if answer["message"] == "Requests rate limit exceeded, please try again later.":
                sleep(21)
                answer = processor(example)
            elif answer["message"] == "The media format is not supported or incorrect for the data inspection.":
                example['image']=[]
                answer=processor(example)
            else:
                exit(-1)
        with open(output, 'a',encoding='UTF-8') as f:
            f.write(json.dumps(answer, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    path = r"./data/all_data.jsonl"
    test(path,r'./subject/Qwen-vl-max.jsonl', process_example_merged_img)