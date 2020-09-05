import matplotlib.pyplot as plt

from utils import read_json_data, aim_list

DATA_PATH = '../result/processed_old_all_info.json'


# def judge_state(result_list):
# 	pos = result_list.count(1)
# 	neg = result_list.count(0)
# 	if pos > 0 :
# 		return 1
# 	elif pos == neg and neg == 0:
# 		return 0
# 	else:
# 		return -1


def draw_one_dot(data, aspect):
    citation = [v['citation'] for k, v in data.items()]
    state = [v['reviews'][aspect] for k, v in data.items()]
    for citation, state, index in zip(citation, state, range(len(state))):
        if citation > 400:
            continue
        if state == 0:
            plt.scatter(index, citation + 1, alpha=0.6, c='red', s=13)
        elif state == 1:
            plt.scatter(index, citation, alpha=0.6, c='green', s=13)

    plt.title(aspect)
    plt.show()


def draw_zhu(data, aspect):
    pass


def gen_bar():
    data = read_json_data(DATA_PATH)
    cite_count = []
    sentiment_count_pos = {}
    sentiment_count_neg = {}
    sentiment_count_all_pos = {}
    sentiment_count_all_neg = {}
    for aim in aim_list:
        sentiment_count_pos[aim] = []
        sentiment_count_neg[aim] = []
        sentiment_count_all_neg[aim] = []
        sentiment_count_all_pos[aim] = []
    upper_limit = 500
    interval = 50
    button_limit = 0
    name_list = range(0, upper_limit + interval, interval)
    for _ in range((upper_limit // interval) + 1):
        cite_count.append(0)
        for aim in aim_list:
            sentiment_count_pos[aim].append(0)
            sentiment_count_neg[aim].append(0)
            sentiment_count_all_neg[aim].append(0)
            sentiment_count_all_pos[aim].append(0)

    for key, val in data.items():
        if val['citation'] > upper_limit or val['citation'] < button_limit:
            continue
        for aim in aim_list:
            if val['reviews'][aim] < 0:
                continue
            if val['reviews'][aim] > 0:
                sentiment_count_pos[aim][val['citation'] // interval] += 1
            else:
                sentiment_count_neg[aim][val['citation'] // interval] += 1
    for aim in aim_list:
        for _ in range((upper_limit // interval) + 1):
            if sentiment_count_pos[aim][_] + sentiment_count_neg[aim][_] != 0:
                sentiment_count_all_pos[aim][_] = sentiment_count_pos[aim][_] / (
                        sentiment_count_pos[aim][_] + sentiment_count_neg[aim][_])
            if sentiment_count_pos[aim][_] + sentiment_count_neg[aim][_] != 0:
                sentiment_count_all_neg[aim][_] = -sentiment_count_neg[aim][_] / (
                        sentiment_count_pos[aim][_] + sentiment_count_neg[aim][_])

    for aim in aim_list:
        plt.bar(range(upper_limit // interval + 1), sentiment_count_all_pos[aim], fc='g', tick_label=name_list)
        plt.bar(range(upper_limit // interval + 1), sentiment_count_all_neg[aim], fc='r', tick_label=name_list)
        plt.title(aim)
        plt.show()


def main():
    gen_bar()


if __name__ == '__main__':
    main()
