#import sys
#sys.path.append('../python_libraries')
from nlp_functions.word_and_character_vectors import PAD_ID,UNK_ID
from nlp_functions.sentence_operations import get_ids_and_vectors
#import word_and_character_vectors
from tqdm import tqdm
import numpy as np
import random
from sklearn.model_selection import train_test_split


class SentimentDataObject(object):
    def __init__(self,word2id,char2id,word_embed_matrix,char_embed_matrix,train_path,test_path,batch_size,review_length,word_length,discard_long=False,test_size=0.1):
        self.one_hot_rating={}
        self.one_hot_rating[0]=[0,0,0,0,1]
        self.one_hot_rating[1]=[0,0,0,1,0]
        self.one_hot_rating[2]=[0,0,1,0,0]
        self.one_hot_rating[3]=[0,1,0,0,0]
        self.one_hot_rating[4]=[1,0,0,0,0]
        train_data_size=156060
        test_data_size=66292
        self.word2id=word2id
        self.char2id=char2id
        self.word_embed_matrix=word_embed_matrix
        self.char_embed_matrix=char_embed_matrix
        self.batch_size=batch_size
        self.review_length=review_length
        self.word_length=word_length
        self.discard_long=discard_long
        train_data=[]
        with open(train_path,'r',encoding='utf-8') as fh:
            for line in tqdm(fh, total=train_data_size+1):
                values=line.split('\t')
                train_data.append((values[2],values[3].strip()))
        train_data.pop(0)
        self.train_data,self.dev_data=train_test_split(train_data,test_size=test_size)

        test_data=[]
        with open(test_path,'r',encoding='utf-8') as fh:
            for line in tqdm(fh,total=test_data_size+1):
                values=line.split('\t')
                test_data.append((values[2],values[0]))
        test_data.pop(0)
        self.test_data=test_data

    def generate_one_epoch(self):
        num_batches = int(len(self.train_data))//self.batch_size
        if self.batch_size*num_batches < len(self.train_data): num_batches += 1
        random.shuffle(self.train_data)

        for i in range(num_batches):
            review_words_for_mask=[]
            review_words=[]
            review_chars=[]
            ratings=[]
            for text,rating in self.train_data[i*self.batch_size:(i+1)*self.batch_size]:
                word_ids, word_ids_to_vectors, char_ids_to_vectors = get_ids_and_vectors(text,self.word2id,self.char2id,self.word_embed_matrix,self.char_embed_matrix,self.review_length,self.word_length,self.discard_long)
                review_words_for_mask.append(word_ids)
                review_words.append(word_ids_to_vectors)
                review_chars.append(char_ids_to_vectors)
                ratings.append(self.one_hot_rating[int(rating)])
            review_words = np.array(review_words)
            review_chars = np.array(review_chars)
            ratings = np.array(ratings)
            review_words_for_mask = np.array(review_words_for_mask)
            review_mask=(review_words_for_mask != PAD_ID).astype(np.int32)
            yield review_words,review_chars,review_mask,ratings

    def generate_dev_data(self):
        num_batches = int(len(self.dev_data))//self.batch_size
        if self.batch_size*num_batches < len(self.dev_data): num_batches += 1
        for i in range(num_batches):
            review_words_for_mask=[]
            review_words=[]
            review_chars=[]
            ratings=[]
            for text,rating in self.train_data[i*self.batch_size:(i+1)*self.batch_size]:
                word_ids, word_ids_to_vectors, char_ids_to_vectors = get_ids_and_vectors(text,self.word2id,self.char2id,self.word_embed_matrix,self.char_embed_matrix,self.review_length,self.word_length,self.discard_long)
                review_words_for_mask.append(word_ids)
                review_words.append(word_ids_to_vectors)
                review_chars.append(char_ids_to_vectors)
                ratings.append(self.one_hot_rating[int(rating)])
            review_words = np.array(review_words)
            review_chars = np.array(review_chars)
            ratings = np.array(ratings)
            review_words_for_mask = np.array(review_words_for_mask)
            review_mask=(review_words_for_mask != PAD_ID).astype(np.int32)
            yield review_words,review_chars,review_mask,ratings

    def generate_test_data(self):
        num_batches=int(len(self.test_data))//self.batch_size
        if self.batch_size*num_batches<len(self.test_data): num_batches += 1
        for i in range(num_batches):
            review_words_for_mask=[]
            review_words=[]
            review_chars=[]
            lineids=[]
            for (text,lineid) in self.test_data[i*self.batch_size:(i+1)*self.batch_size]:
                word_ids, word_ids_to_vectors, char_ids_to_vectors=get_ids_and_vectors(text, self.word2id,self.char2id,self.word_embed_matrix,self.char_embed_matrix,self.review_length,self.word_length,self.discard_long)
                review_words_for_mask.append(word_ids)
                review_words.append(word_ids_to_vectors)
                review_chars.append(char_ids_to_vectors)
                lineids.append(lineid)
            review_words = np.array(review_words)
            review_chars = np.array(review_chars)
            review_words_for_mask = np.array(review_words_for_mask)
            review_mask = (review_words_for_mask != PAD_ID).astype(np.int32)
            lineids=np.array(lineids)
            yield review_words, review_chars, review_mask,lineids

#emb_matrix_char, char2id, id2char=word_and_character_vectors.get_char('C:\\Users\\tihor\\Documents\\ml_data_files')
#emb_matrix_word, word2id, id2word=word_and_character_vectors.get_glove('C:\\Users\\tihor\\Documents\\ml_data_files')
#zz=SentimentDataObject(word2id=word2id,char2id=char2id,word_embed_matrix=emb_matrix_word,char_embed_matrix=emb_matrix_char,train_path='data/train.tsv',test_path='data/test.tsv',batch_size=1000,review_length=52,word_length=15,discard_long=False,test_size=0.1)
#for review_words,review_chars,review_mask,ratings in zz.generate_one_epoch():
#    review_mask_sum = np.sum(review_mask, 1, keepdims=True)
#    review_words_adjusted_batch = np.sum(review_words, 1) / review_mask_sum
#    print(review_words.shape)
#    print(review_chars.shape)
#    print(review_mask.shape)
#    print(ratings.shape)
#    break

#for review_words,review_chars,review_mask in zz.generate_test_data():
#    print(review_words.shape)
#    print(review_chars.shape)
#    print(review_mask.shape)
#    break