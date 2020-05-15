import json
import pickle as pkl

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
