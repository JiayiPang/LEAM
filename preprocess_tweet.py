from gensim.models.word2vec import Word2Vec
import _pickle as cPickle

def precess(filename,encoding,outputfilename): # copy from TWE/preprocess_ch.py
	docno_list=[]
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
			docno_list.append(ix)
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

	vocab1={}                # 过滤次数少的单词
	for w in vocab:
		if vocab[w]>3:
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

	doccententindex = [];
	for doc in doccontent_list:
		docix_list = []
		for w in doc:
			if w in wordtoix:
				docix_list.append(wordtoix[w])
			else:
				docix_list.append(wordtoix['UNK'])
		doccententindex.append(docix_list)
	cPickle.dump([doccententindex,docno_list,wordtoix, ixtoword,doctext_list], open(outputfilename, "wb"))