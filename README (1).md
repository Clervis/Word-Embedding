# Introduction

This package uses word embedding to detect [Domain Generation Algorithms](https://attack.mitre.org/techniques/T1568/002/). Specifically, we're looking for whole word concatenation (cityjulydish.net) and try to determine the relative likelihood that these words would appear together organically. We leverage multiple corpi that have been trained on [Global Vectors for Word Representation (GloVe)](https://nlp.stanford.edu/projects/glove/). 

GloVe is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space.

With this package, you can load in pre-trained vector datasets and determine the semantic distance between words for a relative measure of their relatedness. Word combinations with high distance are more likely to be generated algorithmically.

Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014.

# How to

In a terminal:
```
git clone https://code.levelup.cce.af.mil/netc-dsd/dsc-pgh/word-embedding.git
cd word-embedding
```

In python:
```
from glove import WordEmbedding

WordEmbedding.PreTrainedVectors()
WordEmbedding.ImportWordVectors("glove.6B.50d.txt")

#find distance between two words
WordEmbedding.EuclideanDistance("apple","orange")
> 4.909440040588379

#find distance between vector of word lists
WordEmbedding.LikeAGloVe([["apple","orange","banana"],["master","commander"],["immortal","maiden"]])
> [4.746951262156169, 6.096103668212891, 4.854854583740234]

```

## Pre-Trained Word Vectors

Pre-trained word vectors:
- Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 50d, 100d, 200d, & 300d vectors, 822 MB download): glove.6B.zip
- Common Crawl (42B tokens, 1.9M vocab, uncased, 300d vectors, 1.75 GB download): glove.42B.300d.zip
- Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download): glove.840B.300d.zip
- Twitter (2B tweets, 27B tokens, 1.2M vocab, uncased, 25d, 50d, 100d, & 200d vectors, 1.42 GB download): glove.twitter.27B.zip


This data is made available under the Public Domain Dedication and License v1.0 whose full text can be found at: http://www.opendatacommons.org/licenses/pddl/1.0/.
