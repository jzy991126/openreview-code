import json
import os

from utils import read_json_data, get_sentence_token

RESULT_DIR = '../result'
DATASET_DIR = '../datasets'


def gen_datasets_token():
	result_file = os.path.join(RESULT_DIR, 'datasetv0_token.json')
	data = read_json_data(os.path.join(DATASET_DIR, 'datasetv0.json'))
	for key, val in data.items():
		reviews = val.get('reviews', None)
		if not reviews:
			continue
		res = []
		for review in reviews:
			token = get_sentence_token(review)
			res.append(token)
		data[key]['reviews'] = res
	with open(result_file, 'w') as f:
		json.dump(data, f)


def main():
	gen_datasets_token()


if __name__ == '__main__':
	main()
