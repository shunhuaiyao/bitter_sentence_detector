import notebook
import matplotlib as mpl
import json
from collections import defaultdict
import jieba
import jieba.analyse
import numpy as np
import logging
from gensim.models import word2vec
from gensim import models
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed
from keras.layers import LSTM
from keras.datasets import imdb
from keras.optimizers import Adam


global scores, sentences_seg

def parsePtt():
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	jieba.load_userdict('./data/customDict.txt')
	jieba.analyse.set_stop_words('./data/stopwords.txt')
	with open('./data/lol.json') as data_file:
		data = json.load(data_file)
	stopwordset = set()
	with open('./data/stopwords.txt','r',encoding='utf-8') as sw:
		for line in sw:
			stopwordset.add(line.strip('\n'))

	#data[post_index]["content"]
	#data[post_index]["our_score"]
	#data[post_index]["comments"][comment_index]["content"]
	#data[post_index]["comments"][comment_index]["our_score"]
	#have our scores from 2 to 101
	#3678 posts in total
	global scores, sentences_seg
	scores = []
	sentences_seg = []
	output = open('./data/lol_seg.txt','w')
	for post_index in range(2,3678):
		content_seg = ""
		content_list = jieba.cut(data[post_index]["content"], cut_all=False)
		for content in content_list:
			if content != "\n" and content != " " and content not in stopwordset:
				content_seg += content+" "
		output.write(content_seg)
		output.write("\n")
		if data[post_index].get("our_score"):
			sentences_seg.append(content_seg)
			scores.append(int(data[post_index].get("our_score")))
		for comment_index in range(0,len(data[post_index]["comments"])):
			comment_seg = ""
			comment_list = jieba.cut(data[post_index]["comments"][comment_index]["content"].split(":")[1], cut_all=False)
			for comment in comment_list:
				if comment != "\n" and comment != " " and comment not in stopwordset:
					comment_seg += comment+" "
			output.write(comment_seg)
			output.write("\n")
			if data[post_index]["comments"][comment_index].get("our_score"):
				sentences_seg.append(comment_seg)
				scores.append(int(data[post_index]["comments"][comment_index].get("our_score")))


def  trainWord2Vec():
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	sentences = word2vec.Text8Corpus('./data/lol_seg.txt')
	model = word2vec.Word2Vec(sentences, size=100)
	model.save("./data/lol.model.bin")


def trainLSTM():
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	model = models.Word2Vec.load('./data/lol.model.bin')

	global scores, sentences_seg
	
	for i in range(0, len(sentences_seg)-1):
		num_words = 0
		tmp_sentenceVec = np.zeros(100)
		for word in sentences_seg[i].split(" "):
			if word != "":
				if word in model.wv.vocab:
					num_words = num_words + 1
					tmp_sentenceVec = tmp_sentenceVec + model[word]
		if num_words != 0:
			print(num_words)
			print(scores[i])
			print(tmp_sentenceVec/num_words)




	# TIME_STEPS = 20
	# BATCH_SIZE = 50
	# INPUT_SIZE = 1
	# OUTPUT_SIZE = 1
	# CELL_SIZE = 20
	# LR = 0.006

	# model = Sequential()
	# # build a LSTM RNN
	# model.add(LSTM(
	# 	batch_input_shape=(BATCH_SIZE, TIME_STEPS, INPUT_SIZE),       # Or: input_dim=INPUT_SIZE, input_length=TIME_STEPS,
	# 	output_dim=CELL_SIZE,
	# 	return_sequences=True,      # True: output at all steps. False: output as last step.
	# 	stateful=True,              # True: the final state of batch1 is feed into the initial state of batch2
	# ))
	# # add output layer
	# model.add(TimeDistributed(Dense(OUTPUT_SIZE)))
	# adam = Adam(LR)
	# model.compile(optimizer=adam, loss='mse',)

def get_batch():
	global BATCH_START, TIME_STEPS


if __name__ == "__main__":
	
	parsePtt()
	trainWord2Vec()
	trainLSTM()
