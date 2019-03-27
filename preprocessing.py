import os
import torch
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import time
import os
import copy
import nltk
from nltk.corpus import treebank as treebank


def readTblFromCorpus(corpus):
    tbl = dict()
    tagset = dict()
    for word, tag in corpus.tagged_words():
        if word.lower() not in tbl:
            tbl[word.lower()] = len(tbl)
        if tag not in tagset:
            tagset[tag] = len(tagset)
    return tbl, tagset 


def glove_reader(tbl, path):
    
    glove_dict = dict()
    with open(path) as gloveReader:
        for line in gloveReader:
            parsed_line = line.split()
            glove_dict[parsed_line[0]] = parsed_line[1:]

    pretrained_embedding = torch.zeros(len(tbl)+1, 100)

    for word in tbl:
        if word in glove_dict:
            pretrained_embedding[tbl[word]] = torch.FloatTensor([float(x) for x in glove_dict[word]])
    
    return pretrained_embedding


def divide_data(tagged_sents):
    data = dict()
    maxlength = 0
    
    X = []
    y = []
    for sent in tagged_sents():
        maxlength = max(maxlength, len(sent))
        sent_words = []
        sent_tags = []
        for word in sent:
            sent_words.append(word[0])
            sent_tags.append(word[1])
        X.append(sent_words)
        y.append(sent_tags)
        
    data = (X,y)
            
    return data, maxlength

class tag_dataset(Dataset):
    ''' A dataset for Embedding models '''
    def __init__(self, X, y, tbl, tagset, maxlength, transforms=None):
        self.transforms = transforms

        self.count = len(X)

        combined = [X[i] + [y[i]] for i in range(0, len(X))]
        combined.sort(key=len, reverse=True)

        self.X = []
        self.y = []
        
        for line in combined:
            self.X.append(line[:-1])
            self.y.append(line[-1])

        # Pad sequence with -1 as padding token
        self.X = [torch.LongTensor([tbl[word.lower()] for word in sent]) for sent in self.X]
        self.X = torch.nn.utils.rnn.pad_sequence(self.X, batch_first = True, padding_value = -1)
        
        
        self.y = [torch.LongTensor([tagset[word] for word in sent]) for sent in self.y]
        
        self.y = torch.nn.utils.rnn.pad_sequence(self.y, batch_first = True, padding_value = -1)
        
        
    def __getitem__(self, index):
        return (self.X[index], self.y[index])

    def __len__(self):
        return self.count


def preprocess():
    nltk.download('treebank')
    tbl, tagset = readTblFromCorpus(treebank)
    glove_pretrained = glove_reader(tbl, "glove.6B.300d.txt")

    data, maxlength = divide_data(treebank.tagged_sents)

    data_params = {'training':0.8, 'testing':0.1, 'validation' : 0.1}
    dataloaders = dict()

    datasets = dict()
    datasets['training'] = tag_dataset(data[0][0:int(data_params['training']*len(data[0]))],
     data[1][0:int(data_params['training']*len(data[0]))], tbl, tagset, maxlength)

    datasets['validation'] = tag_dataset(data[0][int(data_params['training']*len(data[0])):-int(data_params['testing']*len(data[0]))],
     data[1][int(data_params['training']*len(data[0])):-int(data_params['testing']*len(data[0]))], tbl, tagset, maxlength)

    datasets['testing'] = tag_dataset(data[0][-int(data_params['testing']*len(data[0])):],
     data[1][-int(data_params['testing']*len(data[0])):], tbl, tagset, maxlength)

    dataset_sizes = dict()
    for phase in ['training','validation','testing']:
        dataloaders[phase] = torch.utils.data.DataLoader(datasets[phase], batch_size=200)
        dataset_sizes[phase] = int(len(datasets) * data_params[phase])



    return glove_pretrained, dataloaders, dataset_sizes, tbl, tagset