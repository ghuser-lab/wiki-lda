from sklearn.cluster import KMeans
import os, sys, numpy, time
import logging, gensim, bz2
import wikipedia
from wiki_helper import *
from bs4 import BeautifulSoup
import requests
import random

n_clusters = int(sys.argv[1])
data_file = sys.argv[2]

data = numpy.loadtxt(data_file)
kmeans = KMeans(init='k-means++', n_clusters=n_clusters)
start = time.time()
kmeans.fit(data)
end = time.time()

print("k-means fitting time: ", end-start)

# start logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# manage file path
corpus_dir = 'corpus'
model_dir = 'model'
prefix = 'simplewiki_'

# load id->word mapping (the dictionary), one of the results of step 2 above
id2word = gensim.corpora.Dictionary.load_from_text(os.path.join(corpus_dir, prefix+'wordids.txt'))

# query user
wikipedia.set_lang("simple")
query_user = 'LittleWink'
print("Query user: ", query_user)
contrib_url = 'https://simple.wikipedia.org/wiki/Special:Contributions/'+query_user

print(contrib_url)
r = requests.get(contrib_url)

# parse contribution page to recommend documents
soup = BeautifulSoup(r.text, "lxml")
changes = soup.find_all('a', {"class" : "mw-changeslist-date"})
if len(changes) == 0:
	# go for next user
	print("next user")

excludes = ['User:', "Talk:", "User talk:", 
            "Category:", "Template:", "Help:"]
recommend = []
for a in changes:
	title = str(a).split('"')[-2]
	if not any(word in title for word in excludes):
		recommend.append(title)

#print(recommend)
query_doc = recommend[0]
print(query_doc)
query_page = wikipedia.page(query_doc)
query_content = query_page.content

# query bag-of-words
tokens = tokenize(query_content)
bow = id2word.doc2bow(tokens)

# load trained LDA model file
lda = gensim.models.LdaModel.load(os.path.join(model_dir, prefix+'lda.model'))
# print(lda[bow])

# use LDA model to predict topic distribution for query doc
K = lda.num_topics
X = numpy.zeros(K, dtype=float)
for topic in lda[bow]:
    X[topic[0]] = topic[1];
print("Topic distribution:")
print(X)

# predict query's cluster
cluster = kmeans.predict(X.reshape(1,-1))   # reshape for predicting 1 sample

# find wiki docs belong to the same cluster
idx = numpy.where(kmeans.labels_==cluster)[0] 
# loop over wiki docs in this cluster to find the smallest Euclidean distance
# between wiki doc and query doc
mindist = 100.
for i in idx:
    dist = numpy.linalg.norm(data[i]-X) 
    if dist < mindist:
        mindist = dist
        minidx = i

print("most similar wiki doc: ", minidx)
print_title(int(minidx))