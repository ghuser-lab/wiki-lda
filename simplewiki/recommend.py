import os, sys, logging, gensim, bz2
from wiki_helper import *
import wikipedia
import requests
import random
from bs4 import BeautifulSoup

# start logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# manage file path
corpus_dir = 'corpus'
model_dir = 'model'
prefix = 'simplewiki_'

# load id->word mapping (the dictionary), one of the results of step 2 above
id2word = gensim.corpora.Dictionary.load_from_text(os.path.join(corpus_dir, prefix+'wordids.txt'))

# load corpus in the Matrix Market format
mm = gensim.corpora.MmCorpus(os.path.join(corpus_dir, prefix+'tfidf.mm'))
# mm = gensim.corpora.MmCorpus(bz2.BZ2File(corpus_dir+'wiki_en_tfidf.mm.bz2')) # use this if you compressed the TFIDF output

# load trained LDA model file
lda = gensim.models.LdaModel.load(os.path.join(model_dir, prefix+'lda.model'))

# per-topic word distribution
# K = lda.num_topics
# topicWordProbMat = lda.print_topics(K)

lang = 'en'
# get query user's categories
wikipedia.set_lang(lang)
query_user = 'User:Adam_Rock'
query_page = wikipedia.WikipediaPage(query_user)
print(query_user, "'s categories:")
print(query_page.categories)

# use the first category of query user
# get category page url
cat = query_page.categories[0].replace(" ", "_")
cat_prefix = 'Category:'

cat_url = 'https://'+lang+'.wikipedia.org/wiki/'+cat_prefix+cat
r = requests.get(cat_url)
print("Choose Category: ", cat)

# parse category page to find users (friends) under the same category
soup = BeautifulSoup(r.text, "lxml")
lis = soup.find_all('li')
friends = []
for li in lis:
    listr = str(li)
    if "/wiki/User:" in listr and query_user not in listr:
        friends.append(listr.split('"')[1].replace("/wiki/User:", ""))

# randomly select a user under this category
# fetch his/her contribution page
friend = random.choice(friends)
contrib_url = 'https://'+lang+'.wikipedia.org/wiki/Special:Contributions/'+friend
print("A Similar user chosen: ", friend)
r = requests.get(contrib_url)

# parse contribution page to recommend documents
soup = BeautifulSoup(r.text, "lxml")
changes = soup.find_all('a', {"class" : "mw-changeslist-date"})
if len(changes) == 0:
	# go for next user
	print("next user")

excludes = ['User:', "Talk:", "User talk:"]
recommend = []
for a in changes:
	title = str(a).split('"')[-2]
	if not any(word in title for word in excludes):
		recommend.append(title)

for topic in recommend:
	print(topic)