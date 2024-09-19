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

    messages = [{'role'   : 'user', 'content': [{'text': '请先一步一步地解决问题，给出最终答案,并将答案填入\"最终答案:\"中:\n已知关于 $x$ 的不等式 $a x^{'
                                                         '2}-x+b \\geq 0$ 的解集为 $[-2,1]$, 则关于 $x$ 的不等式 $b x^{2}-x+a '
                                                         '\\leq 0$ 的解集为 ( )\nA. $[-1,2]$\nB. $\\left[-1, '
                                                         '\\frac{1}{2}\\right]$\nC. $\\left[-\\frac{1}{2}, '
                                                         '1\\right]$\nD. $\\left[\\begin{array}{cc}-1, & -\\frac{1}{'
                                                         '2}\\end{array}\\right]$'}]},
            {'role'   : 'assistant', 'content': [{
                                                  'text': 'C\n$:$ 关于 $\\mathrm{x}$ 的不等式 $\\mathrm{ax}{ }^{'
                                                          '2}-\\mathrm{x}+\\mathrm{b} \\geq 0$ 的解集为 $[-2,1]$,'
                                                          '\n\n$\\therefore-2,1$ 是关于 $\\mathrm{x}$ 的方程 $\\mathrm{'
                                                          'ax}^{2}-\\mathrm{x}+\\mathrm{b}=0$ 的两个根, '
                                                          '$\\therefore\\left\\{\\begin{array}{l}4 a+2+b=0 \\\\ '
                                                          'a-1+b=0\\end{array}\\right.$, 解得 $\\mathrm{a}=-1, '
                                                          '\\mathrm{~b}=2$,\n\n$\\therefore$ 关于 $\\mathrm{x}$ 的不等式 '
                                                          '$\\mathrm{bx}^{2}-\\mathrm{x}+\\mathrm{a} \\leq 0$ 即 $2 '
                                                          '\\mathrm{x}^{2}-\\mathrm{x}-1 \\leq 0$, 解方程 $2 \\mathrm{'
                                                          'x}^{2}-\\mathrm{x}-1=0$, 得 $x_{1}=-\\frac{1}{2}, '
                                                          '\\mathrm{x}_{2}=1$,\n\n$\\therefore$ 关于 $\\mathrm{x}$ 的不等式 '
                                                          '$\\mathrm{bx}^{2}-\\mathrm{x}+\\mathrm{a} \\leq 0$ 的解集为 '
                                                          '$\\left\\{x \\left\\lvert\\,-\\frac{1}{2} \\leq x \\leq '
                                                          '1\\right.\\right\\}$, 即 $\\left[-\\frac{1}{2}, 1\\right]$. '
                                                          '最终答案: C'}]},
            {'role'   : 'user', 'content': [{
                                             'text': '请先一步一步地解决问题，给出最终答案,并将答案填入\"最终答案:\"中:\n已知点 $(-3,-1)$ 和点 $(4,'
                                                     '-6)$ 在直线 $3 x-2 y-a=0$ 的两侧, 则 $a$ 的取值范围为 ( )\nA. $(-24,'
                                                     '7)$\nB. $(-7,24)$\nC. $(-\\infty,-7) \\cup(24, \\infty)$\nD. $('
                                                     '-\\infty,-24) \\cup(7,+\\infty)$'}]},
            {'role'   : 'assistant', 'content': [{
                                                  'text': 'B\n因为点 $(-3,-1)$ 和 $(4,-6)$ 在直线 $3 x-2 y-a=0$ 的两侧, 所以 $[3 \\times(-3)-2 \\times(-1)-a] \\times[3 \\times 4-2 \\times(-6)-a]<0$, 所以 $-7<a<24$. 最终答案: B'}]},
            {'role'   : 'user', 'content': [{
                                             'text': '请先一步一步地解决问题，给出最终答案,并将答案填入\"最终答案:\"中:\n小明在算一道除法时, 把除数 36 错看成了 63 '
                                                     ', 结果得到商是 12 , 正确的商是 ( ）。\nA. 21\nB. 3\nC. 189'}]},
            {'role'   : 'assistant', 'content': [{
                                                  'text': 'A\n答案: A\n\n【分析】把除数 36 错看成了 63 , 结果得到商是 12 , 被除数是不变的, '
                                                          '根据被除数 $=$ 商×除数, 用错误的商乘错误的除数, 计算出被除数, 再用被除数去除以正确的除数, '
                                                          '即可计算出正确的商。据此解\n\n答。\n\n【详解】 $12 \\times 63 \\div '
                                                          '36$\n\n$=756 \\div 36$\n\n$=21$\n\n正确的商是 21 , '
                                                          '选项 $\\mathrm{A}$ 符合题意。\n\n最终答案: '
                                                          'A\n\n【点睛】本题主要考查学生对除数是两位数除法计算方法和被除数、除数与商之间关系的掌握。解决此题的关键是先计算出被除数。'}]},
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
def process_example_merged_img(example, img_folder="./images"):
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
    test(path, path.replace('.jsonl','_qwenvlmax_cot_mergedimg__shot_output.jsonl'), process_example_merged_img)
