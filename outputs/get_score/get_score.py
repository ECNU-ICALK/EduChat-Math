# 将字符串转换为 JSON 对象
import json
import os

def load_jsonl(file_path) -> list:
    with open(file_path, 'r', encoding='UTF-8') as f:
        return [json.loads(line.strip()) for line in f]

def huajian(data):
    del_list = []
    for i in range(12, 0, -1):
        del_list.append('-'+str(i))
        del_list.append('/'+str(i))

    del_list = del_list + ['[',']','*','-']
    # print(del_list)

    data = data.split("\n")[0]
    for i in del_list:
        data = data.replace(i,'')
    return data

# num是数据的第几个，
def chuli(data, num):
    tishi = ['模型A的评分']
    dic = {}
    # 一个模型
    dic[tishi[0]] = data.find(tishi[0])
    dic['结尾'] = len(data)
    # 使用sorted()函数和列表推导式根据值排序
    sorted_d = {k: v for k, v in sorted(dic.items(), key=lambda item: item[1])}         # 用105验证
    # print(sorted_d)

    key_list = []
    value_list = []
    for key, value in sorted_d.items():
        # print(key, value)
        key_list.append(key)
        value_list.append(value)
    print(key_list)
    print(value_list)

    result = []
    idx1 = value_list[0]
    idx2 = value_list[1]
    model = key_list[0]           # 模型
    content = data[idx1:idx2]
    content = huajian(content)
    result.append({"模型": model.replace("的评分",''), "结果": content})

    return result


def tongji(path, save_path):
    data = load_jsonl(path)
    if os.path.exists(save_path):
        os.remove(save_path)
    alld = []
    num = 0
    k = 0
    # 遍历每个 JSON 字符串并解析
    for json_str in data:
        item = json_str
        item_id = item['id']
        content = item['content']
        content = content.replace('#','')
        content = chuli(content, k)
        k += 1
        dic = {"id": item_id, "content": content}
        alld.append(dic)
    with open(save_path, "w", encoding="utf-8") as f:
        for line in alld:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
    print(num)

if __name__ == '__main__':
    for root, dirs, files in os.walk(r'./outputs/score'):
        for file in files:
            if file.endswith('.jsonl'):
                fn_path = os.path.join(root, file)
                save_path = f'./outputs/get_score/{file}'
                tongji(fn_path, save_path)