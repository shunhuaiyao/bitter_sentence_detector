
import json
import random

from collections import defaultdict
import jieba
import jieba.analyse
import numpy as np
import logging
from gensim.models import word2vec
from gensim import models
# from sklearn.feature_extraction import DictVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.svm import LinearSVC
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.datasets import imdb
from keras.optimizers import Adam, RMSprop
from keras.datasets import mnist
from keras.utils import np_utils

global BATCH_START, BATCH_SIZE, INPUT_LEN

def parsePtt():
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	jieba.set_dictionary('./data/dict.txt.big')
	jieba.load_userdict('./data/customDict.txt')
	jieba.analyse.set_stop_words('./data/stopwords.txt')
	with open('./data/lol.json', encoding='utf-8') as data_file:
		data = [json.loads(line) for line in data_file]
	stopwordset = set()
	with open('./data/stopwords.txt','r',encoding='utf-8') as sw:
		for line in sw:
			stopwordset.add(line.strip('\n'))
	# print(len(data))
	# import pdb
	# pdb.set_trace()
	

	#data[post_index]["content"]
	#data[post_index]["our_score"]
	#data[post_index]["comments"][comment_index]["content"]
	#data[post_index]["comments"][comment_index]["our_score"]
	#have our scores from 2 to 101
	#3678 posts in total
	
	#global scores, sentences_seg
	#scores = []
	#sentences_seg = []


	output = open('./data/lol_seg.txt','w', encoding='utf-8')
	training_set = open('./data/training_set.txt', 'w', encoding='utf-8')
	skip = 0
	for post_index in range(2,len(data)):

		content_seg = ""
		content_list = jieba.cut(data[post_index]["content"], cut_all=False)
		for content in content_list:
			if content != "\n" and content != " " and content not in stopwordset:
				content_seg += content+" "
		output.write(content_seg+"\0")
		output.write("\n")
		if data[post_index].get("our_score") and content_seg != "":
			#sentences_seg.append(content_seg)
			#scores.append(int(data[post_index].get("our_score")))
			training_set.write(content_seg+",our_score:"+data[post_index].get("our_score")+"\n")


		for comment_index in range(0,len(data[post_index]["comments"])):
			comment_seg = ""
			try:
				comment_list = jieba.cut(data[post_index]["comments"][comment_index]["content"].split(":")[1], cut_all=False)
			except:
				skip += 1
				continue
			for comment in comment_list:
				if comment != "\n" and comment != " " and comment not in stopwordset:
					comment_seg += comment+" "
			output.write(comment_seg+"\0")
			output.write("\n")
			if data[post_index]["comments"][comment_index].get("our_score") and comment_seg != "":
				#sentences_seg.append(comment_seg)
				#scores.append(int(data[post_index]["comments"][comment_index].get("our_score")))
				training_set.write(comment_seg+",our_score:"+data[post_index]["comments"][comment_index].get("our_score")+"\n")
	print(skip)

def  trainWord2Vec():
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	sentences = word2vec.Text8Corpus('./data/lol_seg.txt')
	model = word2vec.Word2Vec(sentences, size=100)
	model.save("./data/lol.model.bin")



def trainLSTM():
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	
	global bitter, non_bitter, BATCH_START, BATCH_SIZE, INPUT_LEN

	BATCH_SIZE = 50
	BATCH_START = 0
	INPUT_LEN = 5	# 5 words in each sentence
	INPUT_DIM = 250	# each word with 250 dimensions 
	OUTPUT_DIM = 1	# each sentence with a one dimensions output
	CELL_SIZE = 50	# hidden units
	LR = 0.0006

	LSTMmodel = Sequential()
	# build a LSTM RNN
	LSTMmodel.add(LSTM(CELL_SIZE, input_shape=(INPUT_LEN, INPUT_DIM), batch_size=BATCH_SIZE, return_sequences=False, stateful=False))
	#LSTMmodel.add(LSTM(batch_input_shape=(BATCH_SIZE, INPUT_LEN, INPUT_DIM), units=CELL_SIZE, return_sequences=False, stateful=False))
	# add output layer
	LSTMmodel.add(Dense(OUTPUT_DIM))
	adam = Adam(LR)
	LSTMmodel.compile(optimizer=adam, loss='mse',)

	print('Training ------------')
	bitter = 0
	non_bitter = 0
	bitter_sentences_seg = []
	bitter_sentences_dic = {}
	nonbitter_sentences_seg = []
	nonbitter_sentences_dic = {}
	mix_sentences_seg = []
	mix_sentences_dic = {}
	with open('./data/training_baseball_set.txt','r',encoding='utf-8') as lines:
		for line in lines:
			if int(line.strip('\n').split(",our_score:")[1])>1:
				bitter_sentences_seg.append(line.strip('\n').split(",our_score:")[0])
				bitter_sentences_dic[line.strip('\n').split(",our_score:")[0]] = line.strip('\n').split(",our_score:")[1]
			else:
				nonbitter_sentences_seg.append(line.strip('\n').split(",our_score:")[0])
				nonbitter_sentences_dic[line.strip('\n').split(",our_score:")[0]] = line.strip('\n').split(",our_score:")[1]

	## manage data unbalance
	random.shuffle(nonbitter_sentences_seg)
	nonbitter_sentences_seg = nonbitter_sentences_seg[:len(bitter_sentences_seg)]
	mix_sentences_seg = bitter_sentences_seg + nonbitter_sentences_seg
	for x in range(0,len(nonbitter_sentences_seg)):
		mix_sentences_dic[nonbitter_sentences_seg[x]] = nonbitter_sentences_dic[nonbitter_sentences_seg[x]]
	for x in range(0,len(bitter_sentences_seg)):
		mix_sentences_dic[bitter_sentences_seg[x]] = bitter_sentences_dic[bitter_sentences_seg[x]]
	random.shuffle(mix_sentences_seg)

	word2vecModel = models.Word2Vec.load('./data/baseball.model.bin')
	for step in range(len(mix_sentences_seg)):
		if BATCH_START > int(len(mix_sentences_seg)*0.8):
			print("training phase completed")
			break
		X_batch, Y_batch, BATCH_START = get_batch(word2vecModel, mix_sentences_seg, mix_sentences_dic, BATCH_START, BATCH_SIZE, INPUT_LEN)
		cost = LSTMmodel.train_on_batch(X_batch, Y_batch)
		print('train cost: ', cost)

	print('Testing ------------')
	correct = 0
	wrong = 0
	for step in range(len(mix_sentences_seg)):
		X_batch, Y_batch, BATCH_START = get_batch(word2vecModel, mix_sentences_seg, mix_sentences_dic, BATCH_START, BATCH_SIZE, INPUT_LEN)
		if len(X_batch) == 50:
			pred_batch = LSTMmodel.predict(X_batch, BATCH_SIZE)
			for pred_index in range(len(pred_batch)):
				if pred_batch[pred_index][0] * Y_batch[pred_index][0] > 0:
					correct += 1
				else:
					wrong += 1
		else:
			break
	print('bitter:',bitter)
	print('non-bitter',non_bitter)
	
	print('accuracy', correct/(correct+wrong))
	LSTMmodel.save_weights('./data/LSTMmodel_baseball_weight.hdf5')



def get_batch(word2vecModel, mix_sentences_seg, mix_sentences_dic, BATCH_START, BATCH_SIZE, INPUT_LEN):

	global bitter, non_bitter
	X_batch = []
	X_batch_len = 0
	Y_batch = []

	for i in range(BATCH_START, len(mix_sentences_seg)):
		#tmp_sentenceVec = np.zeros(100)
		num_words = 0
		words_list = []
		score = []
		for word in mix_sentences_seg[i].split(" "):
			if word != "" and word in word2vecModel.wv.vocab:
				#tmp_sentenceVec = tmp_sentenceVec + word2vecModel[word]
				if num_words < INPUT_LEN:
					words_list.append(word2vecModel[word])
				num_words = num_words + 1

		if num_words != 0:

			if num_words < INPUT_LEN:	
				for e in range(0, INPUT_LEN - num_words):
					words_list.append(word2vecModel["\0"])
			
			X_batch.append(words_list)
			X_batch_len = X_batch_len + 1
			if int(mix_sentences_dic[mix_sentences_seg[i]])>1:
				bitter = bitter + 1
				score.append(1)
			else:
				non_bitter = non_bitter + 1
				score.append(-1)
			Y_batch.append(score)
			if X_batch_len == BATCH_SIZE:
				BATCH_START = i + 1
				break
			#print(tmp_sentenceVec/num_words)
	#print(len(sentences_seg))
	#print(len(X_batch))
	#print(len(X_batch[0]))
	#print(len(X_batch[0][0]))
	#print(len(Y_batch))
	#print(len(Y_batch[0]))
	#print(BATCH_START)
	return [X_batch, Y_batch, BATCH_START]

def trainNN():

	# data pre-processing
	bitter_sentences_seg = []
	bitter_sentences_dic = {}
	nonbitter_sentences_seg = []
	nonbitter_sentences_dic = {}
	mix_sentences_seg = []
	mix_sentences_dic = {}
	with open('./data/training_lol_set.txt','r',encoding='utf-8') as lines:
		for line in lines:
			if int(line.strip('\n').split(",our_score:")[1])>1:
				bitter_sentences_seg.append(line.strip('\n').split(",our_score:")[0])
				bitter_sentences_dic[line.strip('\n').split(",our_score:")[0]] = 1
			else:
				nonbitter_sentences_seg.append(line.strip('\n').split(",our_score:")[0])
				nonbitter_sentences_dic[line.strip('\n').split(",our_score:")[0]] = 0

	# manage data unbalance
	random.shuffle(nonbitter_sentences_seg)
	nonbitter_sentences_seg = nonbitter_sentences_seg[:len(bitter_sentences_seg)]
	mix_sentences_seg = bitter_sentences_seg + nonbitter_sentences_seg
	for x in range(0,len(nonbitter_sentences_seg)):
		mix_sentences_dic[nonbitter_sentences_seg[x]] = nonbitter_sentences_dic[nonbitter_sentences_seg[x]]
	for x in range(0,len(bitter_sentences_seg)):
		mix_sentences_dic[bitter_sentences_seg[x]] = bitter_sentences_dic[bitter_sentences_seg[x]]
	random.shuffle(mix_sentences_seg)

	# load word2vec model and prepare training, testing data set
	word2vecModel = models.Word2Vec.load('./data/lol.model.bin')
	X_set = []
	Y_set = []
	bitter = 0
	non_bitter = 0
	for x in range(len(mix_sentences_seg)):
		tmp_sentenceVec = np.zeros(250)
		num_words = 0
		for word in mix_sentences_seg[x].split(" "):
			if word != "" and word in word2vecModel.wv.vocab:
				tmp_sentenceVec = tmp_sentenceVec + word2vecModel[word]
				num_words += 1
		if num_words != 0:
			tmp_sentenceVec = tmp_sentenceVec/num_words
			X_set.append(tmp_sentenceVec)
			score = np.zeros(2)
			if mix_sentences_dic[mix_sentences_seg[x]] == 1:
				bitter += 1
				score[0] = 1
			else:
				non_bitter += 1
				score[1] = 1
			Y_set.append(score)
	print('bitter', bitter)
	print('non-bitter', non_bitter)
	# print(len(X_set))
	# print(len(X_set[0]))
	# print(len(Y_set))
	# print(len(Y_set[0]))

	# print(len(X_set[:(int(len(X_set)*0.8))]))
	# print(len(Y_set[:(int(len(Y_set)*0.8))]))

	# print(len(X_set[(int(len(X_set)*0.8)):]))
	# print(len(Y_set[(int(len(Y_set)*0.8)):]))
	

	# build your neural net
	model = Sequential([
	    Dense(32, input_dim=250),
	    Activation('relu'),
	    Dense(2),
	    Activation('softmax'),
	])

	# define your optimizer
	rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

	# add metrics to get more results you want to see
	model.compile(optimizer=rmsprop,
	              loss='categorical_crossentropy',
	              metrics=['accuracy'])

	print('Training ------------')
	# train the model
	X_training_set = np.array(X_set[:(int(len(X_set)*0.8))])
	Y_training_set = np.array(Y_set[:(int(len(Y_set)*0.8))])
	print('train X', len(X_training_set))
	print('train Y', len(Y_training_set))
	model.fit(X_training_set, Y_training_set, epochs=2, batch_size=32)

	print('\nTesting ------------')
	# Evaluate the model with the metrics we defined earlier
	X_testing_set = np.array(X_set[(int(len(X_set)*0.8)):])
	Y_testing_set = np.array(Y_set[(int(len(Y_set)*0.8)):])
	print('test X', len(X_testing_set))
	print('test Y', len(Y_testing_set))
	loss, accuracy = model.evaluate(X_testing_set, Y_testing_set, batch_size=32)

	print('\ntest loss: ', loss)
	print('test accuracy: ', accuracy)




if __name__ == "__main__":
	
	parsePtt()
	trainWord2Vec()
	trainLSTM()
	trainNN()