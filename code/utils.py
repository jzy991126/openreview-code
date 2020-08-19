import json


def read_json_data(data_path):
	with open(data_path) as f:
		data = json.load(f)
	return data


aim_list = ['motivation', 'experiment', 'readable', 'relatework', 'novel']
