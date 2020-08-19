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

ROOT_PATH = os.path.abspath('../')

ELECTRA_SMALL_PATH = os.path.join(ROOT_PATH, 'model/electra-small')
CONFIG_PATH = os.path.join(ELECTRA_SMALL_PATH, 'bert_config_tiny.json')
CHECKPOINT_PATH = os.path.join(ELECTRA_SMALL_PATH, 'electra_small')
DICT_PATH = os.path.join(ELECTRA_SMALL_PATH, 'vocab.txt')

SENTENCE_DATA_PATH = os.path.join(ROOT_PATH, 'datasets/sentence-data')
EXCEL_DATA_PATH = os.path.join(SENTENCE_DATA_PATH, 'predictdata.xlsx')

MODEL_PATH = os.path.join(ROOT_PATH, 'model/sentence-electra-models')
RESULT_PATH = os.path.join('../result')

NUM_CLASSES = 3
MAX_LEN = 256
BATCH_SIZE = 32


def read_excel_data(path, column_name):
	original_data = pd.read_excel(path)
	text_list = original_data[column_name].tolist()
	return_list = []
	for text in text_list:
		if isinstance(text, str):
			return_list.append(str(text).rstrip(string.digits))
	return return_list


class MyDataGenerator(DataGenerator):

	def __init__(self, data, tokenizer, max_len):
		super(MyDataGenerator, self).__init__(data)
		self.tokenizer = tokenizer
		self.max_len = max_len

	def __iter__(self, random=False):
		batch_token_ids, batch_segment_ids = [], []
		for is_end, text in self.sample(random):
			token_ids, segment_ids = self.tokenizer.encode(text, max_length=self.max_len)
			batch_token_ids.append(token_ids)
			batch_segment_ids.append(segment_ids)
			if len(batch_token_ids) == self.batch_size or is_end:
				batch_token_ids = sequence_padding(batch_token_ids)
				batch_segment_ids = sequence_padding(batch_segment_ids)
				yield [batch_token_ids, batch_segment_ids]
				batch_token_ids, batch_segment_ids = [], []


class SentimentPredictor(object):

	def __init__(self, config_path, checkpoint_path, num_classes):
		self.num_classes = num_classes
		self.model = self.__init_model(config_path, checkpoint_path)

	def __init_model(self, config_path, checkpoint_path):
		bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, model='electra',
		                               return_keras_model=False)
		output = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)
		output = Dense(units=self.num_classes, activation='softmax', kernel_initializer=bert.initializer)(output)
		AdamLR = extend_with_piecewise_linear_lr(Adam)
		model = keras.models.Model(bert.model.input, output)
		model.compile(
			loss='sparse_categorical_crossentropy',
			optimizer=AdamLR(learning_rate=1e-3, lr_schedule={1000: 1, 2000: 0.1}),
			metrics=['accuracy'],
		)
		return model

	def load_weights(self, model_path):
		self.model.load_weights(model_path)

	def predict(self, data):
		predict_label = []
		for x_true in data:
			y_pred = self.model.predict(x_true).argmax(axis=1)
			predict_label.append(y_pred)
		return predict_label


def write_result_to_file(predict_result, data, file_path, write_sentences=True):
	with open(file_path, 'w', encoding='utf-8') as f:
		written_sum = 0
		for batch in predict_result:
			for line_result in batch:
				if write_sentences:
					f.write(data[written_sum][0] + '\n')
				f.write(str(line_result) + ' \n')
				written_sum += 1


sentiment_predictor = SentimentPredictor(CONFIG_PATH, CHECKPOINT_PATH, NUM_CLASSES)
tokenizer = Tokenizer(DICT_PATH, do_lower_case=True)


def main(aim):
	original_data = read_excel_data(EXCEL_DATA_PATH, aim)

	data = MyDataGenerator(original_data, tokenizer, MAX_LEN)
	sentiment_predictor.load_weights(os.path.join(MODEL_PATH, aim))
	predict_result = sentiment_predictor.predict(data)

	write_result_to_file(predict_result, original_data, os.path.join(RESULT_PATH, aim + '.txt'))


if __name__ == '__main__':
	# aim_list = ['motivation', 'experiment', 'readable', 'relatework', 'novel']
	aim_list = ['motivation', 'experiment', 'readable', 'relatework', 'novel']
	for aim in aim_list:
		print(aim)
		main(aim)
