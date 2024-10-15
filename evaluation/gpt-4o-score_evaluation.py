import json
import os
import re

# from utils import timestamp, save_jsonl, load_jsonl, find_math_answer, is_equal, is_number

def load_jsonl(path: str):
    with open(path, "r", encoding='utf-8') as fh:
        return [json.loads(line) for line in fh.readlines() if line]


id_raw = {example['id']: example for example in load_jsonl(r"./data/test_data.jsonl")}
tishi = ['模型A']

def find_numbers(s):
    # 使用正则表达式查找所有数字（包括整数和小数）
    numbers = re.findall(r'\d+', s)  # 获取数字
    # numbers = re.findall(r'\d+(\.\d+)?', s)
    return int( numbers[0] )


def get_score(data,model):
    f = 0
    for item in data:
        if f == 1:
            break
        if item['模型'] == model :
            # print(item)
            number = find_numbers(item['结果'])
            f = 1
    # print(number)
    return number

#根据
def evaluate(path,Model,shot):
    results_dict = {}
    data = load_jsonl(path)  # 所有模型输出
    # print(data)
    for example in data:
        id = example['id']
        content = example['content']

        raw_example = id_raw[id]  # 根据id取数据
        subject = raw_example['subject']  # 学科
        level = raw_example['level']  # 年级
        for key in [
            '-all',
            f'-level{level}',
            f'{subject}',
        ]:
            for model in tishi:
                if model not in results_dict:
                    results_dict[model] = {}
                if key not in results_dict[model]:
                    results_dict[model][key] = [0, 0]
                # 对应模型的打分
                number = get_score(content,model)
                # print(number,model)
                results_dict[model][key][0] +=  number
                results_dict[model][key][1] += 1
    # print(results_dict)
    for model in tishi:
        for key in results_dict[model].keys():
            if results_dict[model][key][1] == 0:
                results_dict[model][key] = f'{results_dict[model][key][0]}/{results_dict[model][key][1]}=0'
            else:
                results_dict[model][
                    key] = f'{results_dict[model][key][0]}/{results_dict[model][key][1]}={round(results_dict[model][key][0] / max(results_dict[model][key][1], 1) , 2)}'
    # print(results_dict)
    for model in tishi:
        results_dict[model] = {key: results_dict[model][key] for key in sorted(results_dict[model].keys())}
    # print(results_dict)
    # print(os.path.basename(path), ':\t', results_dict['-all'])
    for i in range(1):
        model = tishi[i]
        name = Model
        if shot == 0 :
            save_path = './evaluation/score_evaluation/' + name + '.txt'
        else:
            save_path = './evaluation/score_evaluation/' + name + '-shot.txt'
        with open (save_path, 'w',encoding='utf-8') as f:
            for key,value in results_dict[model].items():
                f.write(json.dumps((key,value), ensure_ascii=False)+'\n')
    

if __name__ == '__main__':
    for root, dirs, files in os.walk(r'./outputs/get_score'):
        for file in files:
            if file.endswith('.jsonl'):
                fn_path = os.path.join(root, file)
                if fn_path.find('shot') != -1 :
                    evaluate(fn_path,file.replace('.jsonl',''),1)
                else:
                    evaluate(fn_path,file.replace('.jsonl',''),0)