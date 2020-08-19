import json
import os

from tqdm import tqdm

from SentimentPredictor import MyDataGenerator, tokenizer, MAX_LEN, aim_list, sentiment_predictor

DATASETS_PATH = os.path.abspath('../datasets')
JSON_PATH = os.path.join(DATASETS_PATH, 'datasetv0.json')
SENTENCE_MODEL_PATH = os.path.abspath('../model/sentence-electra-models')
RESULT_PATH = '../result/sentiment_analyze.json'


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
	datas = MyDataGenerator(review_list, tokenizer, MAX_LEN)
	result = sentiment_predictor.predict(datas)
	if isinstance(result, list):
		res = []
		for item in result:
			res.extend(item.tolist())
		return res
	return result.tolist()


def main():
	result = {}
	review_data = get_review_data()
	all_data = read_json_data(JSON_PATH)

	length_data = []
	key_data = []
	sentence_data = []
	for key, val in tqdm(review_data.items()):
		length_data.append(len(val))
		key_data.append(key)
		sentence_data.extend(val)
		result[key] = {}
		result[key]['citation'] = all_data[key]['citation']

	for aim in aim_list:
		print(aim)
		sentiment_predictor.load_weights(os.path.join(SENTENCE_MODEL_PATH, aim))
		analyze_result = analyse(sentence_data)
		now_index = 0
		for index in range(len(key_data)):
			key = key_data[index]
			if key not in result:
				result[key] = {}
			result[key][aim] = analyze_result[now_index:now_index + length_data[index]]
			now_index += length_data[index]

	with open(RESULT_PATH, 'w') as f:
		json.dump(result, f)


if __name__ == '__main__':
	main()
