import json
import os
import re

from utils import timestamp, save_jsonl, load_jsonl, find_math_answer, is_equal, is_number

id_raw = {example['id']: example for example in load_jsonl(r"./data/test_data.jsonl")}
Model = ['模型A', '模型B', '模型C', '模型D', '模型E', '模型F',
         '模型G', '模型H', '模型I', '模型J', '模型K', '模型L']
model_name = ['cogvlm_common', 'cogvlm_shot', 'gemini_common', 'Gemini_shot', 'gpt_common', 'gpt_shot',
         'internlm_common', 'internlm_shot', 'mcmd_common', 'mcmd_shot', 'Qwen_common', 'Qwen_shot']
def find_numbers(s):
    numbers = re.findall(r'\d+', s)
    return int( numbers[0] )


def get_score(data,model):
    f = 0
    for item in data:
        if f == 1:
            break
        if item['模型'] == model :
            number = find_numbers(item['结果'])
            f = 1
    return number

def evaluate(path):
    results_dict = {}
    data = load_jsonl(path)
    for example in data:
        id = example['id']
        content = example['content']

        raw_example = id_raw[id]  # 根据id在test_data中取相应数据
        subject = raw_example['subject']  # 学科
        level = raw_example['level']  # 年级
        for key in [
            '-all',
            f'-level{level}',
            f'{subject}',
        ]:
            for model in Model:
                if model not in results_dict:
                    results_dict[model] = {}
                if key not in results_dict[model]:
                    results_dict[model][key] = [0, 0]
                number = get_score(content,model)
                results_dict[model][key][0] +=  number
                results_dict[model][key][1] += 1
    for model in Model:
        for key in results_dict[model].keys():
            if results_dict[model][key][1] == 0:
                results_dict[model][key] = f'{results_dict[model][key][0]}/{results_dict[model][key][1]}=0'
            else:
                results_dict[model][
                    key] = f'{results_dict[model][key][0]}/{results_dict[model][key][1]}={round(results_dict[model][key][0] / max(results_dict[model][key][1], 1) , 2)}'
    for model in Model:
        results_dict[model] = {key: results_dict[model][key] for key in sorted(results_dict[model].keys())}
    for i in range(12):
        model = Model[i]
        name = model_name[i]
        with open (r'./outputs/'+name+'.txt', 'w',encoding='utf-8') as f:
            for key,value in results_dict[model].items():
                f.write(json.dumps((key,value), ensure_ascii=False)+'\n')

if __name__ == '__main__':
    evaluate(r'./outputs/score_model.jsonl')
