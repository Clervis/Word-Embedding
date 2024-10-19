import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import plotly.express as px
import math
import random
from random import sample
import pandas as pd
import itertools
from statistics import mean
from os import walk
import os

class WordEmbedding:

    #def __init__(self, name):
    #    self.name=name

    '''
    This function simply looks in the data folder to determine which pre-trained word vector files are
    available for use and prints their name and filesize.

    Format: ex. glove.twitter.27B.50d.txt is a GloVe vector file with 27 billion tokens, 50 dimensions,
    extracted from tweets.

    Precondition: /data folder contains pretrained vector files
    '''
    def PreTrainedVectors():
        for root, dirs, files in os.walk("./data/"):
            for fn in files:
                path = os.path.join(root, fn)
                size = os.stat(path).st_size
                print(path,f"{round(size/1000000,0):,}",str("MB"))


    '''
    This function loads a vector file into memory. Depending on the filesize, this may take a while.

    Accepts the name of a .txt file, use PreTrainedVectors() to get the name and size.
    Returns nothing.

    Parameter *.txt: the pre-trained word vector file.
    Precondition: the argument is a file in the /data folder
    '''
    def ImportWordVectors(wordvectors):
        global embeddings_dict
        embeddings_dict = {}
        with open(wordvectors, 'r', encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                embeddings_dict[word] = vector

    '''
    This word embedding function, from the pretrained GloVe word vectors of Wikipedia 2014 + Gigaword 5,
    provides a value of relative co-occurence of words.

    Accepts a list of lists of strings (words) [["howdy","doody"],...,["apples","oranges","bananas"]]
    Returns the list of average euclidean distance within sublist [4.3, ...,5.9]

    Parameter WordLists: the list of lists of strings
    Precondition: WordLists is a lists, elements of WordLists are also lists
    '''
    def LikeAGloVe(WordLists):
        scores=[]
        for item in WordLists:
            if len(item)==1:
                scores.append(float('NaN'))
            else:
                permutations=list(itertools.combinations(item,2))
                print(permutations)
                pair_scores=[]
                for pair in permutations:
                    pair_scores.append(spatial.distance.euclidean(embeddings_dict[pair[0]],embeddings_dict[pair[1]]))
                scores.append(mean(pair_scores))
        return scores

    '''
    This word embedding function show linear substructure of two word by finding the euclidean 
    distance between them.

    Accepts two single words
    Returns the euclidean distance between them

    Parameter wordA, wordB
    Precondition: wordA and wordB are in the corpus
    '''
    def EuclideanDistance(wordA, wordB):
        return spatial.distance.euclidean(embeddings_dict[wordA],embeddings_dict[wordB])       
