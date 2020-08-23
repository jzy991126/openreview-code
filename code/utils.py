import json

from nltk.tokenize import sent_tokenize
from nltk.tokenize.punkt import PunktParameters, PunktSentenceTokenizer

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


def replace_specieal_characters(sentences):
	result = []
	for line in sentences:
		result.append(line.replace('+', '').replace('-', ''))
	return result


def split_paragraph_into_sentence(text):
	punkt_param = PunktParameters()
	abbreviation = ['i.e', 'mr', 'st', 'mrs', 'dr', 'ms', 'fig',
	                'u.s.a', 'a.d', 'a.m', 'cap', 'cf', 'cp', 'c.v', 'al'
		, 'etc', 'e.g', 'ff', 'id', 'i.a', 'i.e', 'lb'
		, 'll.b', 'm.a', 'n.b', 'op.cit', 'p.a', 'ph.d'
		, 'p.m', 'p.p', 'prn', 'pro tem', 'p.s', 'q.d'
		, 'q.e.d', 'q.v', 're', 'reg', 'r.i.p', 's.o.s', 'stat'
		, 'vis', 'vs', 'et al', 'et.al', 'etc', 'e.g', 'i.e', 'eq', 'a.e'
		, 'a.e', 'cf', 'con', 'const', 'fig', 's.t', 'st', '(', ')', '?(']
	punkt_param.abbrev_types = set(abbreviation)
	tokenizer = PunktSentenceTokenizer(punkt_param)
	sentences = tokenizer.tokenize(text.lower())
	return replace_specieal_characters(sentences)


def main():
	gen_paper_list()


if __name__ == '__main__':
	main()
