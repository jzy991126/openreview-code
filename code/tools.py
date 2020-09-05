import json
import os

from utils import read_json_data, split_paragraph_into_sentence, aim_list

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
			token = split_paragraph_into_sentence(review)
			res.append(token)
		data[key]['reviews'] = res
	with open(result_file, 'w') as f:
		json.dump(data, f)


def get_sentiment(predicts):
	if len(predicts) == 0:
		return -1
	for sentiment in predicts[::-1]:
		if sentiment != 2:
			return sentiment
	return -1


def get_final_sentiment(sentiments):
	data = []
	for sentiment in sentiments:
		if sentiment >= 0 and sentiment != 2:
			data.append(sentiment)
	if len(data) == 0:
		return -1
	_count = len(set(data))
	if _count == 1:
		return data[0]
	elif _count == 2:
		return 0 if data.count(0) >= data.count(1) else 1
	else:
		return -1


def process_sentiment_data():
	data_path = os.path.join(RESULT_DIR, 'old_all_info.json')
	result_path = os.path.join(RESULT_DIR, 'processed_old_all_info.json')
	res = {}
	data = read_json_data(data_path)
	for key, val in data.items():
		res[key] = {}
		res[key]['citation'] = val['citation']
		res[key]['reviews'] = {}
		for aspect, sentiments in val['reviews'].items():
			temp = []
			for sentiment in sentiments:
				emotion = get_sentiment(sentiment)
				temp.append(emotion)
			res[key]['reviews'][aspect] = get_final_sentiment(temp)
	with open(result_path, 'w') as f:
		json.dump(res, f)


def process_old_sentiment_data():
	data_path = os.path.join(RESULT_DIR, 'old_all_info.json')
	result_path = os.path.join(RESULT_DIR, 'processed_old_all_info.json')
	res = {}
	data = read_json_data(data_path)
	for key, val in data.items():
		res[key] = {}
		res[key]['citation'] = val['citation']
		res[key]['reviews'] = {}
		for aspect, sentiments in val['sentiment'].items():
			temp = []
			for sentiment in sentiments:
				emotion = get_sentiment(sentiment)
				temp.append(emotion)
			res[key]['reviews'][aspect] = get_final_sentiment(temp)
	with open(result_path, 'w') as f:
		json.dump(res, f)


def cal_ave_citation_with_sentiment():
	data_path = os.path.join(RESULT_DIR, 'processed_token_result.json')
	data = read_json_data(data_path)
	res = {}
	for aim in aim_list:
		res[aim] = {}
		res[aim]['citation'] = {}
		res[aim]['count'] = {}
		for _ in range(2):
			res[aim]['citation'][_] = 0
			res[aim]['count'][_] = 0
	for key, val in data.items():
		for aspect, emotion in val['reviews'].items():
			if emotion < 0:
				continue
			res[aspect]['citation'][emotion] += val['citation']
			res[aspect]['count'][emotion] += 1
	print(res)
	for key, val in res.items():
		print(key)
		for _ in range(2):
			if _ == 0:
				print('负面情感平均引用')
			else:
				print('正面情感平均引用')
			if val['count'][_] == 0:
				print('zero count')
			else:
				print(val['citation'][_] / val['count'][_])


def main():
	process_old_sentiment_data()


if __name__ == '__main__':
	main()
