import json
import os

from SentimentPredictor import MyDataGenerator, tokenizer, MAX_LEN, aim_list, sentiment_predictor
from utils import read_json_data

DATASETS_PATH = os.path.abspath('../datasets')
RESULT_PATH = '../result/'
JSON_PATH = os.path.join(RESULT_PATH, 'datasetv0_token.json')
SENTENCE_MODEL_PATH = os.path.abspath('../model/sentence-electra-models')


def get_review_data():
	all_data = read_json_data(JSON_PATH)
	review_data = {}
	for key, val in all_data.items():
		if 'reviews' in val:
			review_data[key] = val['reviews']
	return review_data


def gen_review_processed_data(review_data):
	review_sentences = []
	review_info = []
	for key, reviews in review_data.items():
		for review_index, review in enumerate(reviews):
			for sentence in review:
				review_sentences.append(sentence)
				review_info.append((key, review_index))
	return review_sentences, review_info


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
	review_sentences, review_info = gen_review_processed_data(review_data)
	all_data = read_json_data(JSON_PATH)

	for key, val in review_data.items():
		result[key] = {}
		result[key]['citation'] = all_data[key]['citation']
		result[key]['reviews'] = {}
		for aim in aim_list:
			result[key]['reviews'][aim] = []

	for aim in aim_list:
		print(aim)
		result_label = analyse(review_sentences)
		for predict_result, info in zip(result_label, review_info):
			key, review_index = info[0], info[1]
			if review_index + 1 != len(result[key]['reviews'][aim]):
				result[key]['reviews'][aim].append([])
			result[key]['reviews'][aim][review_index].append(predict_result)

	with open(os.path.join(RESULT_PATH, 'token_result.json'), 'w') as f:
		json.dump(result, f)


if __name__ == '__main__':
	main()
