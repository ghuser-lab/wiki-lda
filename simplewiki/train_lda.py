import sys
import os
import logging
import time
import multiprocessing
import gensim, bz2

# start logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# manage file path
corpus_dir = 'corpus'
prefix = 'simplewiki_'

# load id->word mapping (the dictionary), one of the results of step 2 above
id2word = gensim.corpora.Dictionary.load_from_text(os.path.join(corpus_dir, prefix+'wordids.txt'))

# load corpus in the Matrix Market format
mm = gensim.corpora.MmCorpus(os.path.join(corpus_dir, prefix+'tfidf.mm'))
# mm = gensim.corpora.MmCorpus(bz2.BZ2File(corpus_dir+'wiki_en_tfidf.mm.bz2')) # use this if you compressed the TFIDF output

# print corpus information
# print(mm)

# read from args
K = int(sys.argv[1])
it = int(sys.argv[2])
model_dir = sys.argv[3]

# create dir if necessary
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# max out workers with number of cores
workers = multiprocessing.cpu_count() - 1

# train LDA model
start = time.time()
# serial vs multicore
#lda = gensim.models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=5, update_every=0, passes=1)
lda = gensim.models.LdaMulticore(corpus=mm, id2word=id2word, num_topics=K, passes=it, workers=workers)
end = time.time()
print("time elapsed: ", end-start)

# save model to file
lda.save(os.path.join(model_dir, prefix+'lda.model'))
