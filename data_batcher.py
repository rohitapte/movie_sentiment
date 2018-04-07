from word_and_character_vectors import PAD_ID,UNK_ID
import word_and_character_vectors
from tqdm import tqdm
import numpy as np
import random
from sklearn.model_selection import train_test_split

def split_by_whitespace(sentence):
    words=[]
    for word in sentence.strip().split():
        if word[-1] in '.,?!;':
            words.append(word[:-1])
            words.append(word[-1])
        else:
            words.append(word)
        #words.extend(re.split(" ",word_fragment))
    return [w for w in words if w]

def sentence_to_word_and_char_token_ids(sentence,word2id,char2id):
    """
    convert tokenized sentence into word indices
    any word not present gets converted to unknown
    """
    tokens=split_by_whitespace(sentence)
    word_ids=[]
    char_ids=[]
    for w in tokens:
        word_ids.append(word2id.get(w,UNK_ID))
        char_word_ids=[]
        for c in w:
            char_word_ids.append(char2id.get(c,UNK_ID))
        char_ids.append(char_word_ids)

    return tokens,word_ids,char_ids

def pad_words(word_array,pad_size):
    if len(word_array)<pad_size:
        word_array=word_array+[PAD_ID]*(pad_size-len(word_array))
    return word_array

def pad_characters(char_array,pad_size,word_pad_size):
    if len(char_array)<pad_size:
        char_array=char_array+[[PAD_ID]]*(pad_size-len(char_array))
    for i,item in enumerate(char_array):
        if len(item)<word_pad_size:
            char_array[i]=char_array[i]+[PAD_ID]*(word_pad_size-len(item))
        if len(item) > word_pad_size:
            char_array[i] = item[:word_pad_size]
    return char_array

def convert_ids_to_word_vectors(word_ids,emb_matrix_word):
    retval=[]
    for id in word_ids:
        retval.append(emb_matrix_word[id])
    return retval

def convert_ids_to_char_vectors(char_ids,emb_matrix_char):
    retval=[]
    for word_rows in char_ids:
        row_val=[]
        for c in word_rows:
            row_val.append(emb_matrix_char[c])
        retval.append(row_val)
    return retval

def get_ids_and_vectors(text,word2id,char2id,word_embed_matrix,char_embed_matrix,review_length,word_length,discard_long):
    tokens, word_ids, char_ids = sentence_to_word_and_char_token_ids(text, word2id, char2id)
    if len(tokens) > review_length:
        if discard_long:
            return None,None,None
        else:
            tokens = tokens[:review_length]
            word_ids = word_ids[:review_length]
            char_ids = char_ids[:review_length]
    word_ids = pad_words(word_ids, review_length)
    char_ids = pad_characters(char_ids, review_length, word_length)
    word_ids_to_vectors = convert_ids_to_word_vectors(word_ids, word_embed_matrix)
    char_ids_to_vectors = convert_ids_to_char_vectors(char_ids, char_embed_matrix)
    return word_ids,word_ids_to_vectors,char_ids_to_vectors


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