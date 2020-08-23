import json

from nltk.tokenize import sent_tokenize

aim_list = ['motivation', 'experiment', 'readable', 'relatework', 'novel']

DAVASETV0_PATH = '../datasets/datasetv0.json'


def read_json_data(data_path):
	with open(data_path) as f:
		data = json.load(f)
	return data


def gen_paper_list():
	result_path = '../result/paperlist.json'
	data = read_json_data(DAVASETV0_PATH)
	res = {}
	res['paperlist'] = []
	for key, val in data.items():
		if 'title' in val:
			res['paperlist'].append(val['title'])
	with open(result_path, 'w') as f:
		json.dump(res, f)


def get_sentence_token(sentence):
	sent_tokenize_list = sent_tokenize(sentence)
	return sent_tokenize_list


def main():
	gen_paper_list()


if __name__ == '__main__':
	main()
