import numpy as np
import gensim
import pickle as cPickle

dataset = 'N20short'
if(dataset=='Tweet'):
    loadpath = "./data/langdetect_tweet.p"
    embpath = "./data/langdetect_tweet_emb.p"
    class_name = ['apple','google','microsoft','twitter']
elif(dataset=='N20short'):
    loadpath = './data/N20short.p'
    class_name = ['rec.autos', 'talk.politics.misc', 'sci.electronics', 'comp.sys.ibm.pc.hardware', 'talk.politics.guns',
               'sci.med', 'rec.motorcycles', 'soc.religion.christian', 'comp.sys.mac.hardware', 'comp.graphics',
               'sci.space', 'alt.atheism', 'rec.sport.baseball', 'comp.windows.x', 'talk.religion.misc',
               'comp.os.ms-windows.misc', 'misc.forsale', 'talk.politics.mideast', 'sci.crypt', 'rec.sport.hockey']

x = cPickle.load(open(loadpath, "rb"))
train, val, test = x[0], x[1], x[2]
train_lab, val_lab, test_lab = x[3], x[4], x[5]
wordtoix, ixtoword = x[6], x[7]
print(len(wordtoix))
filename = './data/glove.42B.300d.word2vecformat.txt'
model = gensim.models.KeyedVectors.load_word2vec_format(filename)
vector_size = model.vector_size
embedding_vectors = np.random.uniform(-0.001, 0.001, (len(wordtoix), vector_size))
glove_vocab = list(model.vocab.keys())
count = 0
mis_count = 0
for word in wordtoix.keys():
    idx = wordtoix.get(word)
    if word in glove_vocab:
        embedding_vectors[idx] = model.wv[word]
        count += 1
    else:
        mis_count += 1
print("num of vocab in glove: {}".format(count))
print("num of vocab not in glove: {}".format(mis_count))
cPickle.dump(embedding_vectors, open('./data/langdetect_tweet_emb.p', 'wb'))

