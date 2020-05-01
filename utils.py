import json

def generate_results(predictions):
    with open("data/test-unlabelled.json", 'r') as f:
        test_data = json.load(f)

    output = {}
    index = 0
    for key, val in test_data.items():
        output.update({key: {"label": float(predictions[index])}})
        index += 1

    with open("results/test-output.json", "w") as f:
        f.write(json.dumps(output))


if __name__ == '__main__':
    out_path = 'results/test-output.json'
    with open(out_path, 'r') as f:
        output = json.load(f)

    out_path2 = "D:\\workspace\\jupyter notebook\\COMP90042\\project\\result\\FastText-0.69\\test-output.json"
    with open(out_path2, 'r') as f:
        out_put2 = json.load(f)

    out_path3 = "D:\\workspace\\jupyter notebook\\COMP90042\\project\\result\\FastText-0.66\\test-output.json"
    with open(out_path3, 'r') as f:
        out_put3 = json.load(f)
    
    res = []
    for k, v in output.items():
        res.append(v["label"])

    res2 = []
    for k, v in out_put2.items():
        res2.append(v["label"])
    
    res3 = []
    for k, v in out_put3.items():
        res3.append(v["label"])

    num = 0
    for label in res:
        if label == 1: num += 1
    print(num)

    num = 0
    for label in res2:
        if label == 1: num += 1
    print("0.69", num)

    num = 0
    for label in res3:
        if label == 1: num += 1
    print("0.66: ", num)