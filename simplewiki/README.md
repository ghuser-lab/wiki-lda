### Simple Wiki

1. Download simple wiki dump from https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2.
> only use simplewiki in this tutorial (fewer documents), the full wiki dump is exactly the same format, but larger

2. Use `make_wikicorpus.py` to convert Wiki markup (.xml.bz2 file) to plain text and store the result as sparse TF-IDF vectors:
```
mkdir corpus
python -m gensim.scripts.make_wikicorpus [wiki_dump_name] corpus/simplewiki
```
Corpus files are stored in `corpus` directory with a prefix: `simplewiki`.

3. Train LDA model, you need to specify the number of topics `num_topics`, number of iterations `passes` and output directory in args, using: 
```
python train_lda.py [num_topics] [passes] [model_directory]
```
For example, you may train LDA model with 50 topics and 20 iterations: `python train_lda.py 50 20 model`. The trained LDA model file will be saved in `model` directory. Max number of cores are automatically used.

During training, you will see two parameters:
```
2017-04-24 17:46:58,811 : INFO : topic diff=0.248241, rho=0.200000
```
`rho` is the speed of updating, controlling how much a new chunk influences the result; `topic diff` is how much the topics changed after this EM iteration. No change would indicate that the model has converged.

4. Store document-topic matrix file:
```
python store_data.py model
```
Default name of output file is `docTopicProbMat_[K]`, where K is topic number. File I/O might take some time.

5. Cluster similar wiki doc and Recommend
```
python k-means.py [num_clusters] [data_file]
```

---------------------------

To-do List
