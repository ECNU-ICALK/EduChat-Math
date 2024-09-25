from tqdm import tqdm
import json
from utils import save_jsonl, load_jsonl, find_math_answer, is_equal
import os

# id列表
id_raw = {example['id']: example for example in load_jsonl(r"./data/test_data.jsonl")}


def get_options():
    with open(r'./data/Question_type_index.txt', 'r', encoding='UTF-8') as file:
        data = json.load(file)
    options = data['选择题列表']
    return options


def get_tf():
    with open(r'./data/Question_type_index.txt', 'r', encoding='UTF-8') as file:
        data = json.load(file)
    options = data['判断题列表']
    return options


def evaluate(answer_file, save_path, file, regen_answer=False):
    id_raw1 = {example['id']: example for example in load_jsonl(answer_file)}  #
    alld = []
    options_id = get_options()  # 选择题题号列表
    tf_id = get_tf()

    lines = []  # 存放相应题目的模型答案
    #选择判断放一起
    for id in options_id:
        if id in id_raw:
            lines.append(id_raw1[id])

    for id in tf_id:
        if id in id_raw:
            lines.append(id_raw1[id])

    num = 0
    for line in tqdm(lines, desc=file):
        # print(line)
        raw_exampe = id_raw[line['id']]  # 根据id取测试集数据

        gt_answer = str(raw_exampe['answer'])  # 真实答案

        '''格式化模型输出'''
        if 'model_answer' not in line or regen_answer:  # 没有模型答案 ， 重新格式化
            model_answer = line['answer'].strip()  # 去掉首尾空格
            f = 0
            for c in 'ABCDE':  # 末尾c  (c)  或  开头c\n (c)\n 等答案
                if model_answer.endswith(f"{c}") or model_answer.endswith(f"({c})") or model_answer.startswith(
                        f"{c}\n") or model_answer.startswith(f"({c})\n") or model_answer.startswith(f"({c}) {c}\n"):
                    model_answer = c  # 以ABCDE。开头或结尾，认定其为答案
                    f = 1
            for c in ['对', '错', '√', 'yes', "Yes", "YES", 'False', "FALSE", "false", ]:
                if model_answer.endswith(f"{c}") or model_answer.endswith(f"({c})") or model_answer.startswith(
                        f"{c}\n") or model_answer.startswith(f"({c})\n") or model_answer.startswith(f"({c}) {c}\n"):
                    model_answer = c
                    f = 1
            if f == 0:
                if '[答案]:{' not in model_answer:
                    for flag in ['**答案:**', '[答案]:', '正确答案', '故选', '答案是', '所以', '因此','最终答案']:  # 这些关键词后面是答案
                        raw_model_answer = model_answer  # 复制原答案
                        model_answer = model_answer.split(flag)[-1].strip()  # 原答案变成（存在则为后面一部分，不存在则没操作）
                        if flag in raw_model_answer:  # 存在
                            model_answer = model_answer.split('\n')[0].split('. ')[0]  # 紧跟在提示词后面的一部分
                    model_answer_s = ''
                    for c in "ABCDE":
                        if c in model_answer:
                            model_answer_s = model_answer.join(c)
                    for c in "对错":
                        if c in model_answer:
                            model_answer_s = model_answer.join(c)
                    model_answer = model_answer_s
                elif model_answer.count('[答案]:{') > 1:  # [答案]:{   后面选项的为答案
                    model_answer_v = model_answer.split('[答案]:{')[-1]
                    model_answer = ''
                    for c in "ABCDE":
                        if c in model_answer_v:
                            model_answer = model_answer.join(c)
                    for c in "对错":
                        if c in model_answer_v:
                            model_answer = model_answer.join(c)
            model_answer = find_math_answer(model_answer).replace('(a)', 'a').replace('(b)', 'b').replace('(c)',
                                                                                                          'c').replace(
                '(d)', 'd').replace('(e)', 'e').replace('{a}', 'a').replace('{b}', 'b').replace('{c}', 'c').replace(
                '{d}', 'd').replace('{e}', 'e').rstrip('.').lstrip(':').strip()
            line['model_answer'] = model_answer
        else:
            model_answer = line['model_answer']
        line['correct'] = is_equal(gt_answer, model_answer)
        line['real_answer'] = gt_answer
        if line['correct']:
            num += 1
        alld.append(line)
    save_jsonl(save_path, alld, t_stamp=False)


def math_level_subject_acc(answer_file,file):
    id_raw1 = {example['id']: example for example in load_jsonl(answer_file)}
    options_id = get_options()  # 选择题题号列表
    tf_id = get_tf()

    lines = []  # 存放相应题目的模型答案
    for id in options_id:
        if id in id_raw:
            lines.append(id_raw1[id])

    for id in tf_id:
        if id in id_raw:
            lines.append(id_raw1[id])


    results_dict = {}
    for line in tqdm(lines, desc='math_level_subject_acc'):
        correct = line['correct']
        raw_exampe = id_raw[line['id']]
        subject = raw_exampe['subject']  # 学科
        level = raw_exampe['level']  # 年级
        for key in [
            '-all',
            f'-level{level}',
            f'{subject}',
        ]:
            if key not in results_dict:
                results_dict[key] = [0, 0]
            results_dict[key][0] += 1 if correct else 0
            results_dict[key][1] += 1

    for key in results_dict.keys():
        if results_dict[key][1] == 0:
            results_dict[key] = f'{results_dict[key][0]}/{results_dict[key][1]}=0'
        else:
            results_dict[
                key] = f'{results_dict[key][0]}/{results_dict[key][1]}={round(results_dict[key][0] / max(results_dict[key][1], 1) * 100, 2)}%'

    results_dict = {key: results_dict[key] for key in sorted(results_dict.keys())}
    print(os.path.basename(answer_file), ':\t', results_dict['-all'])
    path = r'./evaluation/acc_evaluate/'
    with open(path + file.replace('.jsonl', '_result.jsonl'), 'w', encoding='utf-8') as f:
        for key, value in results_dict.items():
            f.write(json.dumps((key, value), ensure_ascii=False) + '\n')


if __name__ == '__main__':
    for root, dirs, files in os.walk(r'./outputs/model_answer'):
        for file in files:
            if file.endswith('.jsonl'):
                fn_path = os.path.join(root, file)
                evaluate(fn_path, fn_path, file, True)
                math_level_subject_acc(fn_path,file)
