import json
import os

from SentimentPredictor import MyDataGenerator, tokenizer, MAX_LEN, aim_list, sentiment_predictor
from utils import read_json_data

DATASETS_PATH = os.path.abspath('../datasets')
RESULT_PATH = os.path.abspath('../result/')
JSON_PATH = os.path.join(RESULT_PATH, 'datasetv0_token.json')
SENTENCE_MODEL_PATH = os.path.abspath('../model/sentence-electra-models')


def read_aspect_dict(): # 读取判断是哪个方向的词表
	file_path = os.path.join(DATASETS_PATH, 'sentence-data', 'aspect-data.json')
	aspect_dict = read_json_data(file_path)
	return aspect_dict


def get_aspects(sentence, aspect_dict):
	res = []
	for aim in aim_list:
		for word in aspect_dict[aim]:
			if word in sentence:
				res.append(aim)
	return res


def get_review_data():
	all_data = read_json_data(JSON_PATH)
	review_data = {}
	for key, val in all_data.items():
		if 'reviews' in val:
			review_data[key] = val['reviews']
	return review_data


def gen_review_processed_data(review_data, aspect_dict):
	review_sentences = {}
	review_info = {}
	for aim in aim_list:
		review_sentences[aim] = []
		review_info[aim] = []
	for key, reviews in review_data.items():
		for review_index, review in enumerate(reviews):
			for sentence in review:
				aspects = get_aspects(sentence, aspect_dict)
				for aspect in aspects:
					review_sentences[aspect].append(sentence)
					review_info[aspect].append((key, review_index))
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
	aspect_dict = read_aspect_dict()
	review_sentences, review_info = gen_review_processed_data(review_data, aspect_dict)
	all_data = read_json_data(JSON_PATH)

	for key, val in review_data.items():
		result[key] = {}
		result[key]['citation'] = all_data[key]['citation']
		result[key]['reviews'] = {}
		for aim in aim_list:
			result[key]['reviews'][aim] = []
			for _ in range(len(val)):
				result[key]['reviews'][aim].append([])

	for aim in aim_list:
		print(aim)
		sentiment_predictor.load_weights(os.path.join(SENTENCE_MODEL_PATH, aim))
		result_label = analyse(review_sentences[aim])
		for predict_result, info in zip(result_label, review_info[aim]):
			key, review_index = info[0], info[1]
			result[key]['reviews'][aim][review_index].append(predict_result)

	with open(os.path.join(RESULT_PATH, 'token_result.json'), 'w') as f:
		json.dump(result, f)


if __name__ == '__main__':
	main()
