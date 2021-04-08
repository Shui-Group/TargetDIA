import os
import operator

import numpy as np
import pandas as pd
from scipy import sparse
import torch


class Dictionary(object):
    def __init__(self, aa_list):
        self.word2idx = {}
        self.idx2word = []
        # for padding char:
        self.idx2word.append('*')  # CNN_EMB
        self.word2idx['*'] = 0  # CNN_EMB
        self.build(aa_list)

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def build(self, aa_list):
        for aa in sorted(set(aa_list)):
            self.add_word(aa)
        # print out the dictionary to see:
        for aa in sorted(self.word2idx.items(), key=operator.itemgetter(1)):
            print(aa[1], '->', aa[0])
        print('>> number of aa:', len(self.idx2word))


class Corpus(object):
    def __init__(self, config, dictionary):
        self.config = config

        train_path = config['PATH_TrainSet']
        test_path = config['PATH_TestSet']

        # max length:
        self.max_length = config['Max_Len']
        print('DeepRT: using max length defined by user:', self.max_length)  # DeepRT

        self.dictionary = dictionary

        # Add words to the dictionary, generally there is no new char in test data file, but we still do this again
        seq_data = pd.read_csv(train_path, sep='\t')
        for aa in sorted(set(''.join(seq_data[config['Seq_Col_Name']].values))):
            self.dictionary.add_word(aa)

        # train data:
        self.train, self.train_label, self.train_pepseq = self.tokenize(train_path, pad_length=0)
        print('Read training data done; source:', train_path)
        # self.valid = self.tokenize(os.path.join(path, 'valid.txt'))

        # Test data:
        if '' != test_path:
            self.test, self.test_label, self.test_pepseq = self.tokenize(test_path, pad_length=0)
            print('Read testing data done; source:', test_path)
        else:
            print('Note: didn\'t load test data.' )

        # print out the dictionary to see:
        for aa in sorted(self.dictionary.word2idx.items(), key=operator.itemgetter(1)):
            print(aa[1], '->', aa[0])

    def tokenize(self, path, pad_length=0):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary, generally there is no new char in test data file, but we still do this again
        seq_data = pd.read_csv(path, sep='\t')
        input_size = len(seq_data)
        seq_data = seq_data[seq_data[self.config['Seq_Col_Name']].str.len() <= self.max_length].reset_index(drop=True)
        processed_size = len(seq_data)
        print(f'Dropped {input_size - processed_size} rows caused by the too long peptide length')
        used_pep_seq_list = seq_data[self.config['Seq_Col_Name']].tolist()
        # for aa in sorted(set(''.join(seq_data['sequence'].values))):
        #     self.dictionary.add_word(aa)

        ids = np.zeros((len(seq_data[self.config['Seq_Col_Name']]), self.max_length), dtype=int) # Note: dtype
        # label = np.zeros((len(seq_data['sequence']), max_length)) # the padding value is 0 here, and we just do this as is
        label = np.zeros((len(seq_data[self.config['Seq_Col_Name']]), 1)) # DeepRT
        '''
        num_data = len(seq_data['sequence'])
        ids = np.zeros((num_data*2, self.max_length), dtype=int) # Note: dtype
        # label = np.zeros((len(seq_data['sequence']), max_length)) # the padding value is 0 here, and we just do this as is
        label = np.zeros((num_data*2, 1)) # DeepRT
        '''

        # Tokenize file content
        for index,seq in enumerate(seq_data[self.config['Seq_Col_Name']].values):
            ids[index, -len(seq):] = [self.dictionary.word2idx[aa] for aa in seq] # pad it at the front
            # ids[index+num_data, -len(seq):] = [self.dictionary.word2idx[aa] for aa in seq[::-1]] # data augmentation

        for index,obse in enumerate(seq_data[self.config['RT_Col_Name']].values):
            label[index, 0] = (float(obse)/self.config['Time_Scale']-self.config['Min_RT'])/(self.config['Max_RT']-self.config['Min_RT'])

        ids = torch.LongTensor(ids) # Note: the char index to be embedded has to be int!
        label = torch.FloatTensor(label)

        ids = ids.contiguous()
        label = label.contiguous()

        cuda = False
        if cuda:
            ids = ids.cuda()
            label = label.cuda()

        return ids, label, used_pep_seq_list

