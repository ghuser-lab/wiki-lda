# following the tutorial from:
# http://radimrehurek.com/topic_modeling_tutorial/2%20-%20Topic%20Modeling.html
import itertools
import logging
import gensim
from gensim.utils import smart_open, simple_preprocess
from gensim.corpora.wikicorpus import _extract_pages, filter_wiki
from gensim.parsing.preprocessing import STOPWORDS

def tokenize(text):
    return [token for token in simple_preprocess(text) if token not in STOPWORDS]

def iter_wiki(dump_file):
    """Yield each article from the Wikipedia dump, as a `(title, tokens)` 2-tuple."""
    ignore_namespaces = 'Wikipedia Category File Portal Template MediaWiki User Help Book Draft'.split()
    for title, text, pageid in _extract_pages(smart_open(dump_file)):
        text = filter_wiki(text)
        tokens = tokenize(text)
        if len(tokens) < 50 or any(title.startswith(ns + ':') for ns in ignore_namespaces):
            continue  # ignore short articles and various meta-articles
        yield title, tokens

def print_title(index):
    corpus_dir = 'corpus/'
    dump_file = corpus_dir + 'simplewiki-latest-pages-articles.xml.bz2'
    #stream = iter_wiki(dump_file)
    for title, tokens in itertools.islice(iter_wiki(dump_file), index, index+1):
        print(title, tokens[:10])  # print the article title and its first ten tokens
    return title
