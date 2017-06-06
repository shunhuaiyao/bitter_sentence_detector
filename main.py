
import json
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
from keras.layers import Dense
from keras.layers import LSTM
from keras.datasets import imdb
from keras.optimizers import Adam


global scores, sentences_seg, BATCH_START, BATCH_SIZE, INPUT_LEN

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
		output.write(content_seg+"\0")
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
			output.write(comment_seg+"\0")
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
	
	global BATCH_START, BATCH_SIZE, INPUT_LEN

	BATCH_SIZE = 50
	BATCH_START = 0
	INPUT_LEN = 20	# 20 words in each sentence
	INPUT_DIM = 100	# each word with 100 dimensions 
	OUTPUT_DIM = 1	# each sentence with a one dimensions output
	CELL_SIZE = 50	# hidden units
	LR = 0.006

	LSTMmodel = Sequential()
	# build a LSTM RNN
	LSTMmodel.add(LSTM(CELL_SIZE, input_shape=(INPUT_LEN, INPUT_DIM), batch_size=BATCH_SIZE, return_sequences=False, stateful=False))
	#LSTMmodel.add(LSTM(batch_input_shape=(BATCH_SIZE, INPUT_LEN, INPUT_DIM), units=CELL_SIZE, return_sequences=False, stateful=False))
	# add output layer
	LSTMmodel.add(Dense(OUTPUT_DIM))
	adam = Adam(LR)
	LSTMmodel.compile(optimizer=adam, loss='mse',)

	print('Training ------------')

	word2vecModel = models.Word2Vec.load('./data/lol.model.bin')
	for step in range(501):
		if BATCH_START > 3300:
			break
		X_batch, Y_batch, BATCH_START = get_batch(word2vecModel, BATCH_START, BATCH_SIZE, INPUT_LEN)
		cost = model.train_on_batch(X_batch, Y_batch)
		#pred = model.predict(X_batch, BATCH_SIZE)
		if step % 10 == 0:
		    print('train cost: ', cost)




def get_batch(word2vecModel, BATCH_START, BATCH_SIZE, INPUT_LEN):
	#word2vecModel = models.Word2Vec.load('./data/lol.model.bin')
	global scores, sentences_seg
	
	X_batch = []
	X_batch_len = 0
	Y_batch = []
	
	for i in range(BATCH_START, len(sentences_seg)):
		#tmp_sentenceVec = np.zeros(100)
		num_words = 0
		words_list = []
		score = []
		for word in sentences_seg[i].split(" "):
			if word != "" and word in word2vecModel.wv.vocab:
				#tmp_sentenceVec = tmp_sentenceVec + word2vecModel[word]
				if num_words < INPUT_LEN:
					words_list.append(word2vecModel[word])
				num_words = num_words + 1
		if num_words != 0:
			if num_words < INPUT_LEN:	
				for e in range(0, INPUT_LEN - num_words):
					words_list.append(word2vecModel["\0"])
			if num_words <= INPUT_LEN:
				X_batch.append(words_list)
				X_batch_len = X_batch_len + 1
				score.append(scores[i])
				Y_batch.append(score)
			if X_batch_len == BATCH_SIZE:
				BATCH_START = i + 1
				break
			#print(tmp_sentenceVec/num_words)
	#print(X_batch)
	#print(len(X_batch))
	#print(Y_batch)
	#print(len(Y_batch))
	#print(BATCH_START)
	return [X_batch, Y_batch, BATCH_START]



if __name__ == "__main__":
	
	parsePtt()
	trainWord2Vec()
	trainLSTM()
	#get_batch(74, 50, 20)