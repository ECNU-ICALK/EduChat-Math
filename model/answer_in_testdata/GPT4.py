import base64
import os
import json
import re
import shutil
from time import sleep

import openai
from tqdm import tqdm

os.chdir(os.path.dirname(os.path.abspath(__file__)))

client = openai.OpenAI(api_key="")

MODEL = 'gpt-4o'

def load_jsonl(file_path) -> list:
    with open(file_path, 'r', encoding='UTF-8') as f:
        return [json.loads(line.strip()) for line in f]


def encode_image(image_path):
    if image_path.startswith("http"):
        return image_path
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    print(base64_image)
    return f"data:image/jpeg;base64,{base64_image}"

# Define the function to process each example
def geninput(example):
    question = example['question']
    options = example['options']
    input = '"请先一步一步地解决问题，然后将你的最终答案或一个字母(如果是选择题)放入一个\"[答案]:{}\"中。\ n"' \
            + f'## 题目{question}\n{options}\n'
    return input

def ask_gpt4(question, image_paths):
    content = [{"type": "text", "text": question}]
    for image_path in image_paths:
        content.append({
            "type"     : "image_url",
            "image_url": {
                "url": encode_image(image_path)
            }
        })
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role"   : "user",
                "content": content
            }
        ],
        model=MODEL,
        max_tokens=2048
    )

    return chat_completion.model_dump()


def process_example(example, img_folder='./images'):
    question = geninput(example)
    image_paths = []

    for idx, img in enumerate(example['image']):
        number = re.findall(r'\d+', img)
        number = str(number[0])
        img = number + '.jpg'
        idx += 1
        folder = './images'
        new_path = f"working/gpt4o/{idx}.jpg"

        shutil.copy(os.path.join(img_folder, img), os.path.join(folder, new_path))
        image_paths.append(os.path.join(folder, new_path))
    print(image_paths)
    response = ask_gpt4(question, image_paths)
    response['input'] = question
    response['extra'] = example
    return response


def test(path, output, processor, reset=False):
    if reset or not os.path.exists(output):
        with open(output, 'w', encoding='UTF-8') as f:
            pass
    with open(output, 'r', encoding='UTF-8', errors='replace') as f:
        processed_num = len(f.readlines())
    data = load_jsonl(path)[processed_num:]
    for example in tqdm(data):
        answer = processor(example)
        print(answer)
        while "error" in answer:
            if "Rate limit reached for gpt-4o in organization" in answer["error"]["message"]:
                print(answer["error"]["message"])
                sleep(60)
                answer = processor(example)
            elif answer["error"]["message"] == "Your input image may contain content that is not allowed by our safety system.":
                break
            else:
                print(json.dumps(answer, ensure_ascii=False))
                exit(-1)
        with open(output, 'a',encoding='UTF-8') as f:
            f.write(json.dumps(answer, ensure_ascii=False)+'\n')


if __name__ == '__main__':
    input_path = r'./data/test_data.jsonl'
    output_path = f""
    test(input_path, output_path, process_example)

