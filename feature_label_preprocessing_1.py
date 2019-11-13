
# the label will be two number. Each number refers to the number of months between two concecutive visits. eg [2,3]
# the input csv file should at least contain three columns: 


import os
import math
import random

import torch
import pandas as pd
from datetime import datetime as dt
import datetime as datetime
import numpy as np
import sys



class DataPreprocessing:
    def __init__(self, feature_encoder, batch_size, seq_length=50, test_ratio=0.1,validation_ratio=0.1, shuffle=True):
        self.batch_size = batch_size
        self.feature_encoder = feature_encoder
        self.shuffle = shuffle
        self.validation_ratio = validation_ratio        
        self.seq_length = seq_length

        self.batch_index_train = 0
        self.batch_index_val = 0
        self.batch_index_test = 0

        self.samples = []
        self.read_samples()  
        self.train_samples = []
        self.validation_samples = []
        self.test_samples = []   
        self.split_data()     

    def train_steps(self):
        return math.ceil(len(self.train_samples) / self.batch_size)

    def val_steps(self):
        return math.ceil(len(self.validation_samples) / self.batch_size)

    def test_steps(self):
        return math.ceil(len(self.weekly_samples) / self.batch_size)


    def read_samples(self):

        r = pd.read_csv("data.csv") 
        visit_record = r[r.TEST.str.contains("CD4") | r.TEST.str.contains("VL")]  

        visit_id_date = visit_record[['RFA_ID','TESTDATE']]
        visit_id_date = visit_id_date.drop_duplicates()	

        index = visit_id_date.RFA_ID.drop_duplicates().tolist()

        for i in index:
            s = visit_id_date[visit_id_date.RFA_ID == i]  
            s = s.TESTDATE.tolist()
            if len(s)<3:
               continue
            s = [dt.strptime(x,'%m/%d/%Y') for x in s]
            s.sort()
            self.samples.append((i,[(s[-2]-s[-3]).days/30,(s[-1]-s[-2]).days/30]))       

        if self.shuffle:
            random.shuffle(self.samples)


    def split_data(self):
        """Split training and validation data randomly
        """
        train_val_count = math.ceil(len(self.samples) * (1 - self.test_ratio))
        train_count = train_val_count * (1 - self.validation_ratio)
        self.train_samples = self.samples[:train_count]
        self.validation_samples = self.samples[train_count:train_val_count]
        self.test_samples = self.samples[train_val_count:]


    def batch_train_data(self):
        """A batch of training data
        """
        data = self.batch(self.batch_index_train, self.train_samples)
        self.batch_index_train += 1
        return data

    def batch_val_data(self):
        """A batch of validation data
        """
        data = self.batch(self.batch_index_val, self.validation_samples)
        self.batch_index_val += 1
        return data

    def batch_test_data(self):
        """A batch of test data
        """
        data = self.batch(self.batch_index_test, self.test_samples, testing=True)
        self.batch_index_test += 1
        return data

    def new_epoch(self):
        """New epoch. Reset batch index
        """
        self.batch_index_train = 0
        self.batch_index_val = 0
        self.batch_index_test = 0
     

   def batch(self, batch_index, sample_set, testing=False):
        """Get a batch of samples
        """
        seq_tensors = []

        label_list = []


        def encode_sample(sample):


            label = torch.tensor(sample[1])
            label_list.append(label)

            seq_tensor = self.feature_encoder(sample[0])
            seq_tensors.append(seq_tensor)

        start_i = batch_index * self.batch_size
        end_i = start_i + self.batch_size
        for sample in sample_set[start_i: end_i]:
            encode_sample(sample)

        # in case last batch does not have enough samples, random get from previous samples
        if len(seq_tensors) < self.batch_size:
            for i in random.sample(range(start_i), self.batch_size - len(seq_tensors)):
                encode_sample(sample_set[i])

        return (
                torch.stack(seq_tensors, dim=0),
                torch.stack(label_list,dim=0)
            )



def main():
    test()
    pass

if __name__ == '__main__':
    main()