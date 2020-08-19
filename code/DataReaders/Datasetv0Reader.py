import json
import os

DATASETS_PATH = os.path.abspath('../../datasets')
DATASET_ROOT_PATH = os.path.join(DATASETS_PATH, 'Datasetv0')
DATASET_LIST = os.listdir(DATASET_ROOT_PATH)
OUTPUT_PATH = os.path.join(DATASETS_PATH, 'datasetv0.json')


def my_merge_dict(dict_1, dict_2):
	for key, value in dict_1.items():
		if key not in dict_2:
			dict_2[key] = value
		else:
			dict_2[key].update(value)
	return dict_2


class Davasetv0Reader(object):
	def __init__(self, dataset_root_path):
		self.dataset_root_path = dataset_root_path
		self.dataset_list = [file for file in os.listdir(self.dataset_root_path) if
		                     os.path.isdir(os.path.join(self.dataset_root_path, file))]

	def read_reviews(self, dataset_name):
		table_name = 'reviews'
		dataset_path = os.path.join(self.dataset_root_path, dataset_name)
		review_path = os.path.join(dataset_path, 'reviews')
		review_list = os.listdir(review_path)

		review_dict = {}

		for review_file in review_list:
			id = int(review_file[:-4])
			review_file_path = os.path.join(review_path, review_file)
			with open(review_file_path, encoding='utf-8') as f:
				review_dict[id] = {}
				result = []
				lines = f.readlines()
				for line in lines:
					data = line.strip()
					if len(data) != 0:
						result.append(data)
				review_dict[id][table_name] = result
		return review_dict

	def read_authors(self, dataset_name):
		table_name = 'authors'
		dataset_path = os.path.join(self.dataset_root_path, dataset_name)
		author_file_path = os.path.join(dataset_path, 'authors.txt')
		with open(author_file_path) as f:
			lines = f.readlines()
		author_dict = {}

		id = None
		for line in lines:
			if not id:
				id = int(line)
				author_dict[id] = {table_name: {}}
			else:
				line = line.strip()
				if len(line) == 0:
					id = None
				else:
					info = line.split('\t')
					author_dict[id][table_name][info[0]] = {}
					author_dict[id][table_name][info[0]]['score'] = int(info[1])

		return author_dict

	def read_abstracts(self, dataset_name):
		table_name = 'abstract'
		dataset_path = os.path.join(self.dataset_root_path, dataset_name)
		abstract_file_path = os.path.join(dataset_path, 'abstracts.txt')
		with open(abstract_file_path, encoding='utf-8') as f:
			lines = f.readlines()

		abstract_dict = {}
		for line in lines:
			line = line.strip()
			info = line.split('\t')
			abstract_dict[int(info[0])] = {}
			abstract_dict[int(info[0])][table_name] = info[1]

		return abstract_dict

	def read_citation(self, dataset_name):
		dataset_path = os.path.join(self.dataset_root_path, dataset_name)
		citation_file_path = os.path.join(dataset_path, 'citation.txt')
		citation_dict = {}

		with open(citation_file_path) as f:
			content = f.readlines()
			for line in content:
				line = line.strip()
				info = line.split('\t')
				id = int(info[0])
				citation_dict[id] = {}
				citation_dict[id]['title'] = info[1]
				citation_dict[id]['year'] = int(info[2])
				citation_dict[id]['citation'] = int(info[3])
		return citation_dict

	def read_all_info(self):
		all_info = {}
		for dataset in self.dataset_list:
			all_info = my_merge_dict(self.read_dataset_info(dataset), all_info)
		return all_info

	def read_dataset_info(self, dataset_name):
		info = {}
		info = my_merge_dict(info, self.read_authors(dataset_name))
		info = my_merge_dict(info, self.read_reviews(dataset_name))
		info = my_merge_dict(info, self.read_abstracts(dataset_name))
		info = my_merge_dict(info, self.read_citation(dataset_name))

		return info


def main():
	reader = Davasetv0Reader(DATASET_ROOT_PATH)
	all_info = reader.read_all_info()
	with open(OUTPUT_PATH, 'w') as f:
		json.dump(all_info, f)


# a = reader.read_reviews('ICLR')


if __name__ == '__main__':
	main()
