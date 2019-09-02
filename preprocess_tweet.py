from gensim.models.word2vec import Word2Vec
import _pickle as cPickle
import pandas as pd
import numpy as np
import random
dataset = 'Tweet'
if(dataset=='Tweet'):
    filename = './data/langdetect_tweet.txt'
    labelfilename= './data/langdetect_tweet_label.txt'
    encoding = 'utf-8'
    outputfilename = './data/langdetect_tweet.p'
    loadpath = "./data/langdetect_tweet.p"
    class_name = ['apple','google','microsoft','twitter']

elif(dataset=='N20short'):
	filename = './data/N20short.txt'
	labelfilename= './data/N20short_label.txt'
	encoding = 'utf-8'
	outputfilename = './data/N20short.p'
	loadpath = './data/N20short.p'
	class_name = ['rec.autos', 'talk.politics.misc', 'sci.electronics', 'comp.sys.ibm.pc.hardware', 'talk.politics.guns',
               'sci.med', 'rec.motorcycles', 'soc.religion.christian', 'comp.sys.mac.hardware', 'comp.graphics',
               'sci.space', 'alt.atheism', 'rec.sport.baseball', 'comp.windows.x', 'talk.religion.misc',
               'comp.os.ms-windows.misc', 'misc.forsale', 'talk.politics.mideast', 'sci.crypt', 'rec.sport.hockey']

elif(dataset=='N20small'):
	filename = './data/N20small.txt'
	labelfilename = './data/N20small_label.txt'
	encoding = 'utf-8'
	outputfilename = './data/N20small.p'
	class_name = ['rec.sport.baseball', 'talk.politics.misc', 'comp.windows.x', 'comp.sys.ibm.pc.hardware', 'sci.crypt', 'comp.graphics', 'talk.religion.misc', 'sci.electronics', 'sci.med', 'rec.autos', 'comp.sys.mac.hardware', 'rec.sport.hockey', 'rec.motorcycles', 'sci.space', 'comp.os.ms-windows.misc', 'alt.atheism', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'misc.forsale']


def precess(filename,encoding,outputfilename): # copy from TWE/preprocess_ch.py
	doccontent_list=[]
	doctext_list=[]
	f=open(filename,'r')
	ix=0
	max_len=0
	for line in f:
		doctext_list.append(line)
		token_list=line.split()
		if max_len<len(token_list):
			max_len=len(token_list)
		if len(token_list)!=0:
			doccontent_list.append(token_list)
			ix+=1
	f.close()
	print('docnumber:',ix)
	print('max_len:',max_len)
	vocab={}
	for doc in doccontent_list:
		for w in doc:
			if w in vocab:
				vocab[w]+=1
			else:
				vocab[w]=1

	# name_list = [k.lower().split('.') for k in class_name]
	# id_list = [ [ wordtoix[i] for i in l] for l in name_list]
	# value_list = [ [ opt.W_emb[i] for i in l]    for l in id_list]
	# value_mean = [ np.mean(l)  for l in id_list]

	vocab1={}                # 过滤次数少的单词
	for w in vocab:
		if vocab[w]>2: # todo:参数确定【Tweet语料之前已经过滤了频率少于3的词】
			vocab1[w]=vocab[w]

	print('len_vocab:',len(vocab1))

	wordtoix = {}
	ixtoword = {}
	wordtoix['UNK'] = 0
	wordtoix['END'] = 1
	ixtoword[0] = 'UNK'
	ixtoword[1] = 'END'

	ix = 2 # 从2开始存
	for w in vocab1:
		wordtoix[w] = ix
		ixtoword[ix] = w
		ix += 1

	name_list = [k.lower().split('.') for k in class_name]
	keys = wordtoix.keys()
	class_word_list = []
	for l in name_list:
		for i in l:
			if i in keys:
				continue
			else:
				class_word_list.append(i)
	cix = ix
	class_word_set = list(set(class_word_list))
	for i in class_word_set:
		wordtoix[i] = cix
		ixtoword[cix] = i
		cix += 1
	print(class_word_set)
	print("add class word into vocab, total number:", len(class_word_set))
	# id_list = [[wordtoix[i] for i in l] for l in name_list]
	# value_list = [[W_emb[i] for i in l] for l in id_list]
	# value_mean = [np.mean(l) for l in id_list]

	docs_ix_list = []
	for doc in doccontent_list:
		docix_list = []
		for w in doc:
			if w in wordtoix:
				docix_list.append(wordtoix[w])
			else:
				docix_list.append(wordtoix['UNK'])
		docs_ix_list.append(docix_list)
	return docs_ix_list,wordtoix,ixtoword

def process_label(labelfile):
	df = pd.read_csv(labelfile, names=['label'])
	label_list = df['label'].values.tolist()
	# number_of_labels = len(label_list)

	tmp_list = list(set(label_list))
	tmp_list.sort(key=label_list.index)
	label_dic = dict(zip(tmp_list, list(range(0, len(tmp_list)))))
	# types_of_label = len(label_dic)

	label_key_list = []
	for item in label_list:
		label_key_list.append(label_dic[item])
	return label_key_list,label_dic

def make_train_data(docs_ix_list,label_list,label_dic):
	# 划分数据集
	number_list = []
	for key in label_dic.keys():
		number_list.append(label_list.count(label_dic[key]))

	train = []
	val = []
	test = []
	train_lab = []
	val_lab = []
	test_lab = []

	idx = 0
	# proportion = [0.88, 0.07]
	for num in number_list:
		current_docs_list = docs_ix_list[idx:idx+num]
		current_label_list = label_list[idx:idx+num]

		# 把label转为LEAM用的one-hot格式 e.g. [0 0 1 0]
		for i in range(len(current_label_list)):
			one_digit_label = current_label_list[i]
			tmp = [0] * len(label_dic)
			tmp[one_digit_label] = 1
			current_label_list[i] = np.array(tmp)
		# 取train
		tmp_train = random.sample(current_docs_list, round(num*0.88))
		train +=  tmp_train

		for item in tmp_train:
			idx = current_docs_list.index(item)
			lab = current_label_list.pop(idx)
			train_lab.append(lab)
			current_docs_list.remove(item)

		tmp_val = random.sample(current_docs_list, round(num*0.07))
		val += tmp_val

		for item in tmp_val:
			idx = current_docs_list.index(item)
			lab = current_label_list.pop(idx)
			val_lab.append(lab)
			current_docs_list.remove(item)

		test += current_docs_list
		test_lab += current_label_list
		print(len(train),len(val),len(test))

		idx+=num

	return train,val,test,train_lab,val_lab,test_lab

if __name__ == '__main__':
	docs_ix_list, wordtoix, ixtoword = precess(filename, encoding, outputfilename)
	label_list,label_dic = process_label(labelfilename)
	print(label_dic.keys())
	print(len(wordtoix))
	print(len(ixtoword))

	train, val, test, train_lab, val_lab, test_lab = make_train_data(docs_ix_list, label_list, label_dic)
	cPickle.dump([train, val, test, train_lab, val_lab, test_lab, wordtoix, ixtoword], open(outputfilename, "wb"))
