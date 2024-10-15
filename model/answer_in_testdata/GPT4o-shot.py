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
    # print(base64_image)
    return f"data:image/jpeg;base64,{base64_image}"

# Define the function to process each example
def geninput(example):
    question = example['question']
    options = example['options']
    input = '"请先一步一步地解决问题，给出最终答案,并将答案填入\"最终答案:\"中:\ n"' \
            + f'## 题目{question}\n{options}\n'
    return input

def ask_gpt4(question, image_paths):
    shot = [{'role': 'user', 'content': [{'type': 'text', 'text': '请先一步一步地解决问题，给出最终答案,并将答案填入\"最终答案:\"中:\n已知关于 $x$ 的不等式 $a x^{2}-x+b \\geq 0$ 的解集为 $[-2,1]$, 则关于 $x$ 的不等式 $b x^{2}-x+a \\leq 0$ 的解集为 ( )\nA. $[-1,2]$\nB. $\\left[-1, \\frac{1}{2}\\right]$\nC. $\\left[-\\frac{1}{2}, 1\\right]$\nD. $\\left[\\begin{array}{cc}-1, & -\\frac{1}{2}\\end{array}\\right]$'}]}, {'role': 'assistant', 'content': [{'type': 'text', 'text': 'C\n$:$ 关于 $\\mathrm{x}$ 的不等式 $\\mathrm{ax}{ }^{2}-\\mathrm{x}+\\mathrm{b} \\geq 0$ 的解集为 $[-2,1]$,\n\n$\\therefore-2,1$ 是关于 $\\mathrm{x}$ 的方程 $\\mathrm{ax}^{2}-\\mathrm{x}+\\mathrm{b}=0$ 的两个根, $\\therefore\\left\\{\\begin{array}{l}4 a+2+b=0 \\\\ a-1+b=0\\end{array}\\right.$, 解得 $\\mathrm{a}=-1, \\mathrm{~b}=2$,\n\n$\\therefore$ 关于 $\\mathrm{x}$ 的不等式 $\\mathrm{bx}^{2}-\\mathrm{x}+\\mathrm{a} \\leq 0$ 即 $2 \\mathrm{x}^{2}-\\mathrm{x}-1 \\leq 0$, 解方程 $2 \\mathrm{x}^{2}-\\mathrm{x}-1=0$, 得 $x_{1}=-\\frac{1}{2}, \\mathrm{x}_{2}=1$,\n\n$\\therefore$ 关于 $\\mathrm{x}$ 的不等式 $\\mathrm{bx}^{2}-\\mathrm{x}+\\mathrm{a} \\leq 0$ 的解集为 $\\left\\{x \\left\\lvert\\,-\\frac{1}{2} \\leq x \\leq 1\\right.\\right\\}$, 即 $\\left[-\\frac{1}{2}, 1\\right]$. 最终答案: C'}]},
            {'role': 'user', 'content': [{'type': 'text', 'text': '请先一步一步地解决问题，给出最终答案,并将答案填入\"最终答案:\"中:\n已知点 $(-3,-1)$ 和点 $(4,-6)$ 在直线 $3 x-2 y-a=0$ 的两侧, 则 $a$ 的取值范围为 ( )\nA. $(-24,7)$\nB. $(-7,24)$\nC. $(-\\infty,-7) \\cup(24, \\infty)$\nD. $(-\\infty,-24) \\cup(7,+\\infty)$'}]}, {'role': 'assistant', 'content': [{'type': 'text', 'text': 'B\n因为点 $(-3,-1)$ 和 $(4,-6)$ 在直线 $3 x-2 y-a=0$ 的两侧, 所以 $[3 \\times(-3)-2 \\times(-1)-a] \\times[3 \\times 4-2 \\times(-6)-a]<0$, 所以 $-7<a<24$. 最终答案: B'}]},
            {'role': 'user', 'content': [{'type': 'text', 'text': '请先一步一步地解决问题，给出最终答案,并将答案填入\"最终答案:\"中:\n小明在算一道除法时, 把除数 36 错看成了 63 , 结果得到商是 12 , 正确的商是 ( ）。\nA. 21\nB. 3\nC. 189'}]}, {'role': 'assistant', 'content': [{'type': 'text', 'text': 'A\n答案: A\n\n【分析】把除数 36 错看成了 63 , 结果得到商是 12 , 被除数是不变的, 根据被除数 $=$ 商×除数, 用错误的商乘错误的除数, 计算出被除数, 再用被除数去除以正确的除数, 即可计算出正确的商。据此解\n\n答。\n\n【详解】 $12 \\times 63 \\div 36$\n\n$=756 \\div 36$\n\n$=21$\n\n正确的商是 21 , 选项 $\\mathrm{A}$ 符合题意。\n\n最终答案: A\n\n【点睛】本题主要考查学生对除数是两位数除法计算方法和被除数、除数与商之间关系的掌握。解决此题的关键是先计算出被除数。'}]}]
    if len ( image_paths ) != 0 :
        messages = shot
        content = [{"type": "text", "text": question}]
        for image_path in image_paths:
            content.append({
                "type"     : "image_url",
                "image_url": {
                    "url": encode_image(image_path)
                }
            })
        round_user = {"role": "user", "content": content }
        messages += [round_user]
        print(messages)
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=MODEL,
            max_tokens=2048
        )
    else:
        round_user = {"role": "user", "content": [{"type": "text", "text": question}]}
        messages = shot + [round_user]
        print(messages)
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=MODEL,
            max_tokens=2048
        )
    return chat_completion.model_dump()


def process_example(example, img_folder=r'./images/Test_Images'):
    question = geninput(example)
    image_paths = []

    for idx, img in enumerate(example['image']):
        number = re.findall(r'\d+', img)
        number = str(number[0])
        img = number + '.jpg'
        idx += 1
        folder = r'./images'
        new_path = f"working/{idx}.jpg"
        shutil.copy(os.path.join(img_folder, img), os.path.join(folder, new_path))
        image_paths.append(os.path.join(folder, new_path))
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
    output_path = f"./outputs/{MODEL}-shot_answer.jsonl"
    test(input_path, output_path, process_example)

