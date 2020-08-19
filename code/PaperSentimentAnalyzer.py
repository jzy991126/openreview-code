import json
import os
from tqdm import tqdm

from SentimentPredictor import MyDataGenerator, tokenizer, MAX_LEN, aim_list, sentiment_predictor

DATASETS_PATH = os.path.abspath('../datasets')
JSON_PATH = os.path.join(DATASETS_PATH, 'datasetv0.json')
SENTENCE_MODEL_PATH = os.path.abspath('../model/sentence-electra-models')


def read_json_data(data_path):
	with open(data_path) as f:
		data = json.load(f)
	return data


def get_review_data():
	all_data = read_json_data(JSON_PATH)
	review_data = {}
	for key, val in all_data.items():
		if 'reviews' in val:
			review_data[key] = val['reviews']

	return review_data


def analyse(review_list):
	result_dict = {}
	datas = MyDataGenerator(review_list, tokenizer, MAX_LEN)
	for aim in aim_list:
		sentiment_predictor.load_weights(os.path.join(SENTENCE_MODEL_PATH, aim))
		result = sentiment_predictor.predict(datas)
		result_dict[aim] = result
	return result_dict


def main():
	result = {}
	review_data = get_review_data()
	for key, val in tqdm(review_data.items()):
		result[key] = analyse(val)
	with open('result.json', 'w') as f:
		json.dump(result, f)


if __name__ == '__main__':
	main()