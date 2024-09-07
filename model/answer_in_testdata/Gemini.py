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

image_path_root = "./images"


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

def get_number(data):
    number = re.findall(r'\d+', data)
    number = str(number[0])
    return number


def load_jsonl(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data


# type: text
def get_answer_from_gemini_sample_text(type_, prompt, question, max_tokens=256, temperature=0.0):
    prompt_input = prompt if question == "" else prompt + question if type(prompt) == str else prompt[0] + question + \
                                                                                               prompt[1]
    if type_ == "text":
        response = create_response_gemini_text(prompt_input, max_tokens=max_tokens, temperature=temperature)

        return_txt = ""
        try:
            return_txt = response.text
        except Exception as e:
            print('error', e)
            traceback.print_exc()
            return_txt = str(response)
        return return_txt


# type: vision
def get_answer_from_gemini_sample(type_, prompt, question, images, model, max_tokens=256, temperature=0.0):
    prompt_input = prompt if question == "" else prompt + question if type(prompt) == str else prompt[0] + question + \
                                                                                               prompt[1]
    if type_ == "vision":
        if type(images) == str:
            img = PIL.Image.open(os.path.join(image_path_root, images))
            response = create_response_gemini([prompt_input, img], model=model, max_tokens=max_tokens,
                                              temperature=temperature)
        else:
            ori_paths = []
            for image in images:
                number = get_number(image)
                img = number + '.jpg'
                ori_paths.append(os.path.join(image_path_root, img))
            if prompt_input.find(f'<ImageHere>') != -1:
                paths = ori_paths
            else:
                paths = list(dict.fromkeys(ori_paths))

            imgs = [PIL.Image.open(path) for path in paths]
            new_prompt = []
            tmps = prompt_input.split(f'<ImageHere>')
            for idx, tmp in enumerate(tmps):
                if idx != 0:
                    new_prompt.append(imgs[idx - 1])
                tmp = tmp.strip()

                new_prompt.append(tmp)
            if new_prompt[0] == "":
                new_prompt = new_prompt[1:]
            if new_prompt[-1] == "":
                new_prompt = new_prompt[:-1]
            print("prompt_input:")
            print(new_prompt)
            response = create_response_gemini(new_prompt, model=model, max_tokens=max_tokens, temperature=temperature)
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
        safety_settings=safety_settings,
    )

    return response


def create_response_gemini(messages, model="gemini-pro-vision", max_tokens=256, temperature=0.0, candidate_count=1,
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

    benchmark_prompt = "请先一步一步地解决问题，然后将你的最终答案或一个字母(如果是选择题)放入一个\"[答案]:{}\"中。\ n"

    prompt = benchmark_prompt

    model = 'gemini-1.5-flash'

    max_tokens = 4096

    # 从in_path指定的文件中加载的问题列表。
    questions = load_jsonl(in_path)[:]
    last_BEGIN = -1
    retry_num = 0
    max_retry_num = 10


    old_id = []

    while True:
        try:
            all = load_jsonl(save_path)  # 之前训练过的数据
        except FileNotFoundError:
            all = []

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

        counter = BEGIN

        try:
            for idx, line in enumerate(tqdm(questions[BEGIN:END])):
                if line['id'] not in old_id:
                    question = line['question']
                    options = line['options']
                    question = f"## 题目 {question}\n{options}"
                    if line['question'].find('<ImageHere>') == -1 and str( line['options'] ) .find('<ImageHere>')==-1:
                        response = get_answer_from_gemini_sample_text('text', prompt, question,
                                                                 max_tokens=max_tokens, temperature=0.0)
                    else:
                        response = get_answer_from_gemini_sample('vision', prompt, question, line['image'], model,
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
            traceback.print_exc()

        save_jsonl(all, save_path, mode='w', add_timestamp=False, verbose=False)


if __name__ == '__main__':
    parser = ArgumentParser(description="A simple argument parser")
    parser.add_argument("--in_path", type=str, help="input path of data",
                        default=r'')
    parser.add_argument("--save_path", type=str, help="save path of model outputs",
                        default=r'')
    args = parser.parse_args()
    benchmark_gemini(in_path=args.in_path, save_path=args.save_path)
