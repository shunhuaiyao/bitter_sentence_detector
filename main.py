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

	output = open('./data/lol_seg.txt','w')
	for post_index in range(2,3678):
		content_seg = ""
		content_list = jieba.cut(data[post_index]["content"], cut_all=False)
		for content in content_list:
			if content != "\n" and content != " " and content not in stopwordset:
				content_seg += content+" "
		output.write(content_seg)
		output.write("\n")
		for comment_index in range(0,len(data[post_index]["comments"])):
			comment_seg = ""
			comment_list = jieba.cut(data[post_index]["comments"][comment_index]["content"].split(":")[1], cut_all=False)
			for comment in comment_list:
				if comment != "\n" and comment != " " and comment not in stopwordset:
					comment_seg += comment+" "
			output.write(comment_seg)
			output.write("\n")

def  trainWord2Vec():
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	sentences = word2vec.Text8Corpus('./data/lol_seg.txt')
	model = word2vec.Word2Vec(sentences, size=100)
	model.save("./data/lol.model.bin")
	

def trainLSTM():
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	model = models.Word2Vec.load('./data/lol.model.bin')
	print(model["台灣"])
	print(model.similarity("台灣", "巴西"))


if __name__ == "__main__":
	parsePtt()
	trainWord2Vec()
	trainLSTM()