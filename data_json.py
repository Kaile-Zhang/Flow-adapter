import json, random


def data1():
    json_path = '/data/Hszhu/dataset/PIE-Bench_v1/generated_dataset_full_pack.json'
    data = json.load(open(json_path, 'r', encoding='utf-8'))
    result = list()
    for i in data.keys():
        images = data[i]['instances']['0']
        for j in images.keys():
            image = images[j]
            result.append({
                "targ_image": image['gen_img_path'],
                "text": image['edit_prompt'],
                "orig_image": image['ori_img_path']
            })

    result_json = open('data.json', 'w', encoding='utf-8')
    random.shuffle(result)
    json.dump(result, result_json, ensure_ascii=False)

def data2():
    # json_path = '/data/Hszhu/dataset/Gedi_v1/annotations.json'
    json_path = 'annotations.json'
    data = json.load(open(json_path, 'r', encoding='utf-8'))
    result = list()
    for i in data.keys():
        di = data[i]
        for j in di.keys():
            dj = di[j]
            for k in dj.keys():
                result.append(dj[k])

    print(len(result))
    result_json = open('data.json', 'w', encoding='utf-8')
    random.shuffle(result)
    json.dump(result, result_json, ensure_ascii=False)

def data3():
    json_path = '/data/Hszhu/dataset/Gedi_v1/annotations.json'
    # json_path = 'annotations.json'
    data = json.load(open(json_path, 'r', encoding='utf-8'))
    result = list()
    for i in data.keys():
        di = data[i]
        for j in di.keys():
            dj = di[j]
            for k in dj.keys():
                result.append(dj[k]['ori_img_path'])
                result.append(dj[k]['gen_img_path'])

    print(len(result))
    result = list(tuple(result))
    print(len(result))
    result_json = open('data_rec.json', 'w', encoding='utf-8')
    random.shuffle(result)
    json.dump(result, result_json, ensure_ascii=False)

def data4():
    json_path = '/data/Hszhu/dataset/Gedi_v1/annotations.json'
    # json_path = 'annotations.json'
    data = json.load(open(json_path, 'r', encoding='utf-8'))
    result = list()
    for i in data.keys():
        di = data[i]
        for j in di.keys():
            dj = di[j]
            for k in dj.keys():
                image = dj[k]
                result.append(image)

    print(len(result))
    random.shuffle(result)
    result_json = open('data_train.json', 'w', encoding='utf-8')
    json.dump(result, result_json, ensure_ascii=False)

def data5():
    json_path = '/data/Hszhu/dataset/Gedi_full/annotations.json'
    # json_path = 'annotations.json'
    data = json.load(open(json_path, 'r', encoding='utf-8'))
    result = list()
    for i in data.keys():
        di = data[i]
        for j in di.keys():
            dj = di[j]
            for k in dj.keys():
                image = dj[k]
                result.append(image)

    print(len(result))
    random.shuffle(result)
    result_json = open('data_full.json', 'w', encoding='utf-8')
    json.dump(result, result_json, ensure_ascii=False)



data5()
