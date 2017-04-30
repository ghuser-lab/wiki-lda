import os, sys, numpy, logging, gensim, bz2

# manage file path
corpus_dir = 'corpus'
prefix = 'simplewiki_'
model_dir = sys.argv[1]

# load corpus in the Matrix Market format
mm = gensim.corpora.MmCorpus(os.path.join(corpus_dir, prefix+'tfidf.mm'))
# mm = gensim.corpora.MmCorpus(bz2.BZ2File(corpus_dir+'wiki_en_tfidf.mm.bz2')) # use this if you compressed the TFIDF output

# load trained LDA model file
lda = gensim.models.LdaModel.load(os.path.join(model_dir, prefix+'lda.model'))

# per-topic word distribution
K = lda.num_topics
topicWordProbMat = lda.print_topics(K)

# store per-doc topic distribution
data = numpy.zeros((mm.__len__(), K), dtype=float)
for i, doc in enumerate(mm):
  for topic in lda[doc]:
    data[i, topic[0]] = topic[1];

# save data file
numpy.savetxt(os.path.join(model_dir, 'docTopicProbMat_'+str(K)),
              data, fmt='%1.4f')
