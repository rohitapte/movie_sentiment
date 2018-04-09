import pandas as pd
from sklearn import preprocessing
import numpy as np
import csv
from sklearn.model_selection import train_test_split

#glove_file='../glove/glove.6B.300d.txt'
#pretrained_vectors=pd.read_table(glove_file, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
#base_vector=pretrained_vectors.loc['this'].as_matrix()

def vectorize_sentence(sentence,character_mappings,length):
    x=[]
    for w in sentence.lower():
        x.append(character_mappings[w])
    if len(x)<length:
        zero_mapping=[0 for i in range(len(x[0]))]
        for i in range(length-len(x)):
            x.append(zero_mapping)
    return x

class SentimentCharacterCNNObject(object):
    def __init__(self, test_ratio=0.1):
        self.df_train_input=pd.read_csv('data/train.tsv', sep='\t')
        self.df_test_input=pd.read_csv('data/test.tsv', sep='\t')
        sentences_train=self.df_train_input['Phrase'].tolist()
        sentences_test=self.df_test_input['Phrase'].tolist()
        text=''
        maxlen=0
        for sentence in sentences_train:
            if len(sentence)>maxlen:
                maxlen=len(sentence)
            for s in sentence.lower():
                text+=s
        for sentence in sentences_test:
            if len(sentence)>maxlen:
                maxlen=len(sentence)
            for s in sentence.lower():
                text+=s
        chars=set(text)
        vocab_size=len(chars)
        lb=preprocessing.LabelBinarizer()
        lb.fit(list(chars))
        temp=lb.transform(list(chars))
        character_mappings={}
        for i in range(len(lb.classes_)):
            character_mappings[lb.classes_[i]]=temp[i].tolist()
        train_temp=[]
        for item in sentences_train:
            train_temp.append(vectorize_sentence(item,character_mappings,maxlen))
        test_temp=[]
        for item in sentences_test:
            test_temp.append(vectorize_sentence(item,character_mappings,maxlen))
        train_labels=self.df_train_input['Sentiment'].tolist()
        unique_labels=list(set(train_labels))
        lb=preprocessing.LabelBinarizer()
        lb.fit(unique_labels)
        self.y_data=lb.transform(train_labels)
        self.train_data=np.array(train_temp)
        self.test_data=np.array(test_temp)
        self.X_train,self.X_cv,self.y_train,self.y_cv=train_test_split(self.train_data,self.y_data,test_size=test_ratio)

    def generate_one_epoch(self,batch_size=100):
        num_batches=int(self.X_train.shape[0])//batch_size
        if batch_size*num_batches<self.X_train.shape[0]:
            num_batches+=1
        perm=np.arange(self.X_train.shape[0])
        np.random.shuffle(perm)
        self.X_train=self.X_train[perm]
        self.y_train=self.y_train[perm]
        for j in range(num_batches):
            batch_X=self.X_train[j*batch_size:(j+1)*batch_size]
            batch_y=self.y_train[j*batch_size:(j+1)*batch_size]
            yield batch_X,batch_y