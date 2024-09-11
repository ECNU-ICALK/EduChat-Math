import base64
import os
import json
import re
import shutil
from time import sleep
import requests

import openai
from tqdm import tqdm
import concurrent.futures

os.chdir(os.path.dirname(os.path.abspath(__file__)))

'''
用Gpt4o对模型结果进行打分



'''
def load_jsonl(file_path) -> list:
    with open(file_path, 'r', encoding='UTF-8') as f:
        return [json.loads(line.strip()) for line in f]


client = openai.OpenAI(api_key="")

MODEL = 'gpt-4o'


def get_list_id():
    with open(r'./data/Question_type_index.txt', 'r', encoding='UTF-8') as file:
        data = json.load(file)
    options = data['解答题列表']
    options = options + data['填空题列表']  # 全部的解答填空题
    return options


# Define the function to process each example
def geninput(example, model_answer_path):
    id_raw = {example['id']: example for example in load_jsonl(model_answer_path)}
    id = example['id']
    question = example['question']
    options = example['options']
    answer = example['answer']
    analysis = example['analysis']
    ansA = id_raw[id]['answer']
    input = '"请作为一名严格、公正的数学裁判，对这个语言模型（模型A）给出的中文数学问题解答进行综合评估。评估标准为答案的准确性、完整性、逻辑性以及对问题的理解深度。 你将获得标准答案与解析，以及模型A的解答。"' \
            + '请注意：客观公正： 避免因模型名称、回答长度或其他主观因素影响评分。全面评估： 比较模型答案与标准答案，并分析原因。' \
            + '避免偏见： 不应提前设定对模型的偏好。格式规范： 请严格按照以下格式输出最终评分：模型A的评分：[1-10分]' \
            + f"输入格式为:\n[问题]{question}\n{options}\n[标准答案]{answer}\n[答案解析]{analysis}\n" \
            + f"[模型A的输出]:{ansA}" \
            + f"输出格式为：模型A的评分:[1-10分]"
    return input


def ask_gpt4(question):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role"   : "user",
                "content": question
            }
        ],
        model=MODEL,
        max_tokens=2048
    )
    return chat_completion.model_dump()


def process_example(example, model_answer_path):
    question = geninput(example, model_answer_path)
    response = ask_gpt4(question)
    response['input'] = question
    response['extra'] = example
    return response


'''
找出测试集中的解答填空题,当做test数据集,保存在lines
'''
def test(path, output,model_answer_path, processor, reset=False):
    id_raw1 = {example['id']: example for example in load_jsonl(path)}
    list_id = get_list_id()
    lines = []
    for id in list_id:
        if id in id_raw1:
            lines.append(id_raw1[id])
    #print(len(lines))
    if reset or not os.path.exists(output):
        with open(output, 'w', encoding='UTF-8') as f:
            pass
    with open(output, 'r', encoding='UTF-8', errors='replace') as f:
        processed_num = len(f.readlines())
    data = lines[processed_num:]

    for example in tqdm(data):
        answer = processor(example, model_answer_path)
        print(answer)
        with open(output, 'a', encoding='UTF-8') as f:
            f.write(json.dumps(answer, ensure_ascii=False) + '\n')
        assert answer['choices'][0]['finish_reason'] in ['stop', 'length']


if __name__ == '__main__':
    for root, dirs, files in os.walk(r'./outputs/model_answer'):
        for file in files:
            if file.endswith('.jsonl'):
                fn_path = os.path.join(root, file)
                test(r'./data/test_data.jsonl', fn_path.replace('model_answer', 'Gpt4o_score'),
                     fn_path, process_example)
