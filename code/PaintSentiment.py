import math

import matplotlib.pyplot as plt

from utils import read_json_data

DATA_PATH = '../result/sentiment_analyze.json'


def judge_state(result_list):
	pos = result_list.count(1)
	neg = result_list.count(0)
	if pos > neg:
		return 1
	elif pos == neg:
		return 0
	else:
		return -1


def draw_one(data, aspect, pos):
	plt.subplots(pos)
	citation = [v['citation'] for k, v in data.items()]
	state = [judge_state(v[aspect]) for k, v in data.items()]
	for citation, state, index in zip(citation, state, range(len(state))):
		if state == 0:
			plt.scatter(index, math.log2(citation + 1), alpha=0.6, c='red', s=13)
		elif state == 1:
			plt.scatter(index, math.log2(citation + 1), alpha=0.6, c='green', s=13)
	# else:
	# 	plt.scatter(index, math.log2(citation + 1), alpha=0.6, c='gray', s=13)
	plt.title(aspect)
	plt.show()


def main():
	data = read_json_data(DATA_PATH)
	from utils import aim_list
	pos = 0

	for aim in aim_list:
		draw_one(data, aim, pos)
		pos += 1


if __name__ == '__main__':
	main()
