import pandas as pd
import numpy as np
import csv
#from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.tokenize import RegexpTokenizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

glove_file='../glove/glove.6B.300d.txt'
pretrained_vectors=pd.read_table(glove_file, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
base_vector=pretrained_vectors.loc['this'].as_matrix()
def vec(w):
    try:
        location=pretrained_vectors.loc[w]
        return location.as_matrix()
    except KeyError:
        return None

def get_average_vector(review):
    numwords=0.0001
    average=np.zeros(base_vector.shape)
    tokenizer = RegexpTokenizer(r'\w+')
    for word in tokenizer.tokenize(review):
    #sentences=sent_tokenize(review)
    #for sentence in sentences:
    #    for word in word_tokenize(sentence):
        value=vec(word.lower())
        if value is not None:
            average+=value
            numwords+=1
        #else:
        #    print("cant find "+word)

    average/=numwords
    return average.tolist()

def vectorize_sentence(sentence,char_indices,length):
    X=[]
    x=[char_indices[w] for w in sentence.lower()]
    if len(X)<length:
        x.extend([0 for i in range(length-len(x))])
    x2=np.eye(len(char_indices))[x]
    X.append(x2)
    return x2.reshape((1,x2.shape[0],x2.shape[1]))

class SentimentDataObject(object):
    def __init__(self,test_ratio=0.1):
        self.df_train_input=pd.read_csv('data/train.tsv',sep='\t')
        self.df_test_input=pd.read_csv('data/test.tsv',sep='\t')
        self.df_train_input['Vectorized_review']=self.df_train_input['Phrase'].apply(lambda x:get_average_vector(x))
        self.df_test_input['Vectorized_review'] = self.df_test_input['Phrase'].apply(lambda x: get_average_vector(x))
        self.train_data=np.array(self.df_train_input['Vectorized_review'].tolist())
        self.test_data=np.array(self.df_test_input['Vectorized_review'].tolist())
        train_labels=self.df_train_input['Sentiment'].tolist()
        unique_labels=list(set(train_labels))
        self.lb=LabelBinarizer()
        self.lb.fit(unique_labels)
        self.y_data=self.lb.transform(train_labels)
        self.X_train,self.X_cv,self.y_train,self.y_cv=train_test_split(self.train_data,self.y_data,test_size=test_ratio)

        sentences=self.df_train['Phrase'].tolist()
        sentences.extend(self.df_test['Phrase'].tolist())
        text=''
        self.maxlen=0
        for sentence in sentences:
            if len(sentence)>self.maxlen:
                self.maxlen=len(sentence)
            for s in sentence.lower():
                text+=s
        chars = set(text)
        self.vocab_size=len(chars)
        self.char_indices=dict((c, i) for i, c in enumerate(chars))
        self.indices_char=dict((i, c) for i, c in enumerate(chars))
        train_phrases=self.df_train['Phrase'].tolist()
        test_phrases=self.df_test['Phrase'].tolist()
        self.X_train_CNN=np.zeros(shape=(0,self.maxlen,self.vocab_size))



    def generate_one_epoch_for_neural(self,batch_size=100):
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

