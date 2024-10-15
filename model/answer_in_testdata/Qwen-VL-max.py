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
    input = '"请先一步一步地解决问题，给出最终答案,并将答案填入\"最终答案:\"中:\ n"' \
            + f'## 题目{question}\n{options}\n'
    return input


def load_jsonl(file_path) -> list:
    with open(file_path, 'r', encoding='UTF-8') as f:
        return [json.loads(line.strip()) for line in f]


def ask_qwvl(question, image_paths):
    content = []
    i = 0
    for image_path in image_paths:
        if (i <= 9):
            content.append({
                "image": f"{image_path}"
            })
        i += 1
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
def process_example_merged_img(example, img_folder="./images/Test_Images"):
    input = geninput(example)

    image_paths = []
    for s in example['image']:
        number = re.findall(r'\d+', s)
        number = str(number[0])
        img = number + '.jpg'
        image_paths.append(os.path.join(img_folder, img))
    answer = ask_qwvl(input, image_paths)
    answer['extra'] = example
    return answer


def test(path, output, processor, reset=False):
    if reset or not os.path.exists(output):
        with open(output, 'w', encoding='UTF-8') as f:
            pass
    with open(output, 'r', encoding='UTF-8') as f:
        processed_num = len(f.readlines())
    data = load_jsonl(path)[processed_num:]
    for example in tqdm(data):
        answer = processor(example)
        while answer["status_code"] != HTTPStatus.OK:
            if answer["message"] == "Requests rate limit exceeded, please try again later.":
                sleep(21)
                answer = processor(example)
            elif answer["message"] == "The media format is not supported or incorrect for the data inspection.":
                example['image'] = []
                answer = processor(example)
            else:
                exit(-1)
        with open(output, 'a', encoding='UTF-8') as f:
            f.write(json.dumps(answer, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    path = r"./data/test_data.jsonl"
    output = r'./outputs/qwen-vl-max.jsonl'
    test(path, output, process_example_merged_img)
