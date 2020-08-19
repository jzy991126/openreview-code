import os
import string

import pandas as pd
from bert4keras.backend import keras, set_gelu
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.tokenizers import Tokenizer
from keras.layers import Lambda, Dense

set_gelu('tanh')

config_path = os.path.abspath('electra-small/bert_config_tiny.json')
checkpoint_path = os.path.abspath('electra-small/electra_small')
dict_path = os.path.abspath('electra-small/vocab.txt')
data_path = os.path.abspath('sentence-data/predictdata.xlsx')
model_path = os.path.abspath('model')
result_path = os.path.abspath('result')

tokenizer = Tokenizer(dict_path, do_lower_case=True)

num_classes = 3
max_len = 256
batch_size = 32


def read_message(path, column_name):
	original_data = pd.read_excel(path)
	text_list = original_data[column_name].tolist()
	return_list = []
	for text in text_list:
		return_list.append([str(text).rstrip(string.digits), 0])

	return return_list


def evaluate(data, model):
	total, right = 0., 0.
	for x_true, y_true in data:
		y_pred = model.predict(x_true).argmax(axis=1)
		y_true = y_true[:, 0]
		total += len(y_true)
		right += (y_true == y_pred).sum()
	return right / total


def predict(data, model):
	predict_label = []
	for x_true, y_true in data:
		y_pred = model.predict(x_true).argmax(axis=1)
		predict_label.append(y_pred)
	return predict_label


# import sys
# sys.exit()

class data_generator(DataGenerator):
	def __iter__(self, random=False):
		batch_token_ids, batch_segment_ids, batch_labels = [], [], []
		for is_end, (text, label) in self.sample(random):
			token_ids, segment_ids = tokenizer.encode(text, max_length=max_len)
			batch_token_ids.append(token_ids)
			batch_segment_ids.append(segment_ids)
			batch_labels.append([label])
			if len(batch_token_ids) == self.batch_size or is_end:
				batch_token_ids = sequence_padding(batch_token_ids)
				batch_segment_ids = sequence_padding(batch_segment_ids)
				batch_labels = sequence_padding(batch_labels)
				yield [batch_token_ids, batch_segment_ids], batch_labels
				batch_token_ids, batch_segment_ids, batch_labels = [], [], []


def main(aim):
	bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, model='electra',
	                               return_keras_model=False, )

	# change the .csv file to you own data is ok. and also change line 121 to restore the predicated results.
	predict_data = read_message(data_path, aim)

	print(predict_data)
	output = Lambda(lambda x: x[:, 0],
	                name='CLS-token')(bert.model.output)
	output = Dense(units=num_classes,
	               activation='softmax',
	               kernel_initializer=bert.initializer)(output)

	model = keras.models.Model(bert.model.input, output)
	model.summary()
	AdamLR = extend_with_piecewise_linear_lr(Adam)

	model.compile(
		loss='sparse_categorical_crossentropy',
		# optimizer=Adam(2e-3),
		optimizer=AdamLR(learning_rate=1e-3,
		                 lr_schedule={1000: 1, 2000: 0.1}),
		metrics=['accuracy'],
	)

	pre_predict = data_generator(predict_data, batch_size)

	model.load_weights(os.path.join(model_path, aim))
	# 预测结果
	predict_result = predict(pre_predict, model)
	# 验证模型
	# you can change the restore files.
	with open(os.path.join(result_path, aim + '.txt'), 'w', encoding='utf-8') as f:
		for i in predict_result:
			for j in i:
				f.write(str(j) + '\n')


# print(j)
# print(predict_result)
# evaluate_result = evaluate(pre_predict, model)
# print(u'final test acc: %05f\n' % (evaluate(pre_predict, model)))


if __name__ == '__main__':
	aim_list = ['novel', 'readable', 'motivation', 'experiment', 'writing']
	for aim in aim_list:
		print(aim)
		main(aim)
