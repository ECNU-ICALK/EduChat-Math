import csv
import random
import re

import google.generativeai as genai
from time import sleep
import requests
import base64
import json
import time
import PIL.Image
import os
import pprint
from argparse import ArgumentParser
import traceback
from tqdm import tqdm

os.chdir(os.path.dirname(os.path.abspath(__file__)))

GOOGLE_API_KEY = ""
genai.configure(api_key=GOOGLE_API_KEY, transport='rest')

rows = []

def timestamp() -> str:
    nowtime = time.strftime('-%Y%m%d-%H%M', time.localtime(time.time()))
    print(nowtime)
    return nowtime


def save_jsonl(data: list, path: str, mode='w', add_timestamp=True, verbose=True) -> None:
    if add_timestamp:
        file_name = f"{path.replace('.jsonl', '')}{timestamp()}.jsonl"
    else:
        file_name = path
    with open(file_name, mode, encoding='utf-8') as f:
        if verbose:
            for line in tqdm(data, desc='save'):
                f.write(json.dumps(line, ensure_ascii=False) + '\n')
        else:
            for line in data:
                f.write(json.dumps(line, ensure_ascii=False) + '\n')

def load_jsonl(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data

def get_number(data):
    number = re.findall(r'\d+', data)
    number = str(number[0])
    return number


# 类型: 纯文本
def get_answer_from_gemini_sample_text(type_, prompt, question, max_tokens=256, temperature=0.0):
    prompt_input = prompt if question == "" else prompt + question if type(prompt) == str else prompt[0] + question + \
                                                                                               prompt[1]
    if type_ == "text":
        # 获得模型答案
        response = create_response_gemini_text(prompt_input, max_tokens=max_tokens, temperature=temperature)

        return_txt = ""
        try:
            return_txt = response.text
        except Exception as e:
            print('error', e)
            traceback.print_exc()
            return_txt = str(response)
        return return_txt



def create_response_gemini_text(messages, model="gemini-pro", max_tokens=256, temperature=0.0, candidate_count=1,
                                stop_sequences=None):
    safety_settings = [
        {
            "category" : "HARM_CATEGORY_DANGEROUS",
            "threshold": "BLOCK_NONE",
        },
        {
            "category" : "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category" : "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
        {
            "category" : "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category" : "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE",
        },
    ]

    model = genai.GenerativeModel(model)

    response = model.generate_content(
        messages,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
            candidate_count=candidate_count,
            stop_sequences=stop_sequences
        ),
        safety_settings=safety_settings
    )

    return response


def benchmark_gemini(in_path, save_path):
    class_prompt = '请基于以下## 题目 内容，为题目选择一个合适的知识点:\n' \
            '## 注意1. 知识点的类别只能是固定的【解析几何】【算术】【组合几何学】' \
                                             '【组合数学】【计数】【画法几何学】【图论】【逻辑题】【度量几何角】【度量几何面积】' \
                                             '【度量几何长度】【立体几何学】【统计数学】【拓扑学】【变换几何】【代数】十六个种类中的一类，' \
                                             '**绝对不能有其他额外的类别\n**2. 除此以外不要生成其他任何额外内容\n' \
                                             '3. 只是进行知识点分类"'
    prompt = class_prompt

    model = 'gemini-1.txt.5-flash'

    max_tokens = 4096

    questions = load_jsonl(in_path)[:]

    last_BEGIN = -1
    retry_num = 0
    max_retry_num = 10


    old_id = []

    while True:

        try:
            all = load_jsonl(save_path)  # 之前训练的数据
        except FileNotFoundError:
            all = []  # 之前没有训练过数据

        BEGIN = len(all)

        if BEGIN == last_BEGIN:
            retry_num += 1
            print(save_path, f'ERROR: BEGIN == last_BEGIN {last_BEGIN}, retry_num {retry_num}')
            if retry_num > max_retry_num:
                print(save_path, f'ERROR: retry_num > max_retry_num {max_retry_num}, BEGIN {BEGIN}')
                break
        else:
            retry_num = 0
            last_BEGIN = BEGIN

        END = len(questions)
        if BEGIN >= END:
            if BEGIN > END:
                print(save_path, f'ERROR: BEGIN {BEGIN} > END {END}')
            else:
                print(save_path, 'DONE', END)
            break
        print(save_path, f'BEGIN: {BEGIN}, END: {END}')
        outs = []

        counter = BEGIN  # 记录之前记录了多少个

        try:
            for idx, line in enumerate(tqdm(questions[BEGIN:END])):
                if line['id'] not in old_id:
                    question = line['question']
                    options = line['options']
                    question = f"## 题目 {question}\n{options}"
                    response = get_answer_from_gemini_sample_text('text', prompt, question,
                                                                 max_tokens=max_tokens, temperature=0.0)

                    print(response)
                    res = {'response': response, 'system': prompt, 'model': model, 'extra': line.copy()}
                outs.append(res)
                all.append(res)
                counter += 1
                if counter == 3 or counter % 10 == 0 or counter == END:
                    save_jsonl(outs, save_path, mode='a', add_timestamp=False, verbose=False)
                    outs = []

        except Exception as e:
            sleep(20)
            print('error', e)
            traceback.print_exc()  # 打印异常信息和堆栈跟踪

        save_jsonl(all, save_path, mode='w', add_timestamp=False, verbose=False)


if __name__ == '__main__':
    parser = ArgumentParser(description="A simple argument parser")
    parser.add_argument("--in_path", type=str, help="input path of data",
                        default = r'./data/all_data.jsonl')                                               # 原始数据
    parser.add_argument("--save_path", type=str, help="save path of model outputs",
                        default=r'./subject/gemini.jsonl')
    args = parser.parse_args()
    benchmark_gemini(in_path=args.in_path, save_path=args.save_path)
