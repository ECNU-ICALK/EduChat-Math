import time
import os
import json
import openai
from multiprocessing import Pool
from tqdm import tqdm
os.chdir(os.path.dirname(os.path.abspath(__file__)))


client = openai.OpenAI(api_key = "")

MODEL = 'gpt-4o'

def load_jsonl(file_path)->list:
    with open(file_path, 'r',encoding='UTF-8') as f:
        return [json.loads(line.strip()) for line in f]

# Define the function to process each example
def geninput(example):
    question = example['question']
    options = example['options']
    input = '请基于以下## 题目 内容，对题目进行知识点的分类:\n' \
            ''+f'## 题目{question}\n{options}\n'+'## 注意1. 知识点的类别只能是固定的【解析几何】【算术】【组合几何学】' \
                                             '【组合数学】【计数】【画法几何学】【图论】【逻辑题】【度量几何角】【度量几何面积】' \
                                             '【度量几何长度】【立体几何学】【统计数学】【拓扑学】【变换几何】【代数】十六个种类中的一类，' \
                                             '**绝对不能有其他额外的类别\n**2. 请以## 本体知识点 开头完成分类，除此以外不要生成其他任何额外内容\n' \
                                             '3. 只是进行知识点分类，**不需要对题目进行解答**4.输出格式是"本题知识点为："'

    return input


def ask_gpt4(question):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": question
            }
        ],
        model=MODEL,
        max_tokens=2048
    )
    return chat_completion.model_dump()

def process_example(example):
    question = geninput(example)
    for i in range(15):
        question = question.replace(f'<ImageHere>', '').strip()
    response = ask_gpt4(question)
    response['input'] = question
    response['extra'] = example
    return response

def test(path, output, processor, reset=False):
    if reset or not os.path.exists(output):
        with open(output, 'w',encoding='UTF-8') as f:
            pass
    with open(output, 'r',encoding='UTF-8',errors='replace') as f:
        processed_num = len(f.readlines())
    data = load_jsonl(path)[processed_num:]
    for example in tqdm(data):
        answer = processor(example)
        print(answer)
        with open(output, 'a',encoding='UTF-8') as f:
            f.write(json.dumps(answer, ensure_ascii=False)+'\n')
        assert answer['choices'][0]['finish_reason'] in ['stop', 'length']



if __name__ == '__main__':
    input_path = r"./data/all_data.jsonl"
    output_path = f"./subject/gpt4o.jsonl"
    test(input_path, output_path, process_example)
