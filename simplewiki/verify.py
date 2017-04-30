import os, sys, logging, gensim, bz2
from wiki_helper import *
import wikipedia

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
#print(mm[0])
#print(mm)
# load trained LDA model file
lda = gensim.models.LdaModel.load(os.path.join(model_dir, prefix+'lda.model'))

# per-topic word distribution
K = lda.num_topics
topicWordProbMat = lda.print_topics(K)

# query document
# use doc_id to verify lda model
doc_id = int(sys.argv[1])    # select a document idx within simplewiki dump

# use wiki wrapper to grab the latest version from current simple wiki
wikipedia.set_lang("simple")
query_page = wikipedia.WikipediaPage(print_title(doc_id))
query_doc = query_page.content
#print(query_page.categories)

# query bag-of-words
tokens = tokenize(query_doc)
bow = id2word.doc2bow(tokens)


# per-doc topic distribution
# docTopicProMat is a list of tuples
# docTopicProMat[m][n] gives you the n-th topic's pencentage in the m-th document
# ex: print(docTopicProbMat[5378][4])
docTopicProbMat = lda[mm]
print("dump doc: ", docTopicProbMat[doc_id])
print("fresh doc: ", lda[bow])
# how to retrieve a document
# for word_id, freq in mm[707]:
    # print(id2word[word_id], freq)
#print_title(doc_id)
print(query_doc)
