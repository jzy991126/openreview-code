import json
import pickle

from bert_serving.client import BertClient
from tqdm import tqdm

from utils import read_json_data

# bc = BertClient(ip='219.216.65.104')  # ip address of the GPU machine


def gen_vector():
    source_file = '../result/datasetv0_token.json'
    aim_file = '../result/datasetv0-vector.json'
    data = read_json_data(source_file)
    for key, content in tqdm(data.items()):
        content['vector'] = []
        if 'reviews' not in content:
            continue
        for tokens in content['reviews']:
            temp = []
            for token in tokens:
                if token != '':
                    temp.append(token)
            res = [bc.encode(temp).tolist()]
            content['vector'].append(res)
    with open(aim_file, 'w') as f:
        json.dump(data, f)


def collect_info():
    source_file = '../result/datasetv0-vector.json'
    res_file = '../result/vectors.pkl'
    data = read_json_data(source_file)
    res = []
    for key, content in tqdm(data.items()):
        if 'vector' not in content:
            continue
        for vec in content['vector']:
            for v in vec:
                res.append(v)
    with open(res_file, 'wb') as f:
        pickle.dump(res, f)


def main():
    collect_info()


if __name__ == '__main__':
    main()
