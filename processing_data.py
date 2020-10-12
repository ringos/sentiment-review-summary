import os
import random
import copy
import torch
import torch.nn as nn
import torch.utils.data as tdata
import gzip
import pickle
import csv
from nltk.tokenize import sent_tokenize
import time
import numpy as np
from main import seed
import glob

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

BERT_MAX_SEQ_LEN=512


class MixedDataset(tdata.Dataset):
    '''
    under the condition of using BERT, we would get token embedding inside the model, 
    so we do not need to get token embeddings here
    '''
    def __init__(self,corpus_ids,labels,embedding,sum_ids=None):
        super(MixedDataset,self).__init__()
        self.embedding=embedding
        if sum_ids==None:
            self.data=[corpus_ids,labels]
        else:
            self.data=[corpus_ids,labels,sum_ids]
    def __getitem__(self,index):
        sample=dict()
        sample['raw']=self.embedding(torch.LongTensor(self.data[0][index]))
        sample['label']=self.data[1][index]
        if len(self.data)==3:
            sample['sum']=self.embedding(torch.LongTensor(self.data[2][index]))
        return sample
    def __len__(self):
        return len(self.data[1])

def batchify(batch):
    raw=[batch[i]['raw'] for i in range(len(batch))]
    
    sum_flag=0
    
    if 'sum' in batch[0]:
        sum_flag=1
        sum=[batch[i]['sum'] for i in range(len(batch))]
    labels=[batch[i]['label'] for i in range(len(batch))]
    
    raw_lengths = list(map(len,raw))
    max_raw_len=max(raw_lengths)
    raw_lengths = torch.tensor(raw_lengths)

    padded_raw=nn.utils.rnn.pad_sequence(raw,batch_first=True)
    
    if sum_flag==1:
        sum_lengths=list(map(len,sum))
        max_sum_len=max(sum_lengths)
        sum_lengths=torch.tensor(sum_lengths)
        padded_sum=nn.utils.rnn.pad_sequence(sum,batch_first=True)

    labels = torch.tensor(labels)

    if sum_flag==1:
        return [padded_raw,raw_lengths,max_raw_len],labels,[padded_sum,sum_lengths,max_sum_len]
    return [padded_raw,raw_lengths,max_raw_len],labels,False



def load_txt_folder(folder_path,begin=None,end=None):
    print(folder_path)
    file_list = glob.glob(os.path.join(os.getcwd(), folder_path, "*.txt"))
    corpus = []
    for file_path in file_list:
        with open(file_path) as f_input:
            corpus.append(copy.deepcopy(f_input.read()))
    # print('internal len {}'.format(len(corpus)))
    return corpus

def load_dataset(param, tokenizer, dataset_name, mode, usage='train', decode_folder=None, output_att_weight=False, use_shuffle=True):
    print('\n-----Loading dataset-----')
    load_time=time.time()
    file_path=param.FILE_PATH
    if mode=='golden':
        if dataset_name=='toy':
            tmp_data=extract_raw_sum(file_path+param.TOY_FILE)
        elif dataset_name=='sports':
            tmp_data=extract_raw_sum(file_path+param.SPORTS_FILE)
        elif dataset_name=='movie':
            tmp_data=extract_raw_sum(file_path+param.MOVIE_FILE)
        else:
            raise ValueError('The input dataset_name \'{}\' does\'nt exist'.format(dataset_name))
    elif mode=='predicted':
        if dataset_name=='toy':
            tmp_data=extract_raw_sum(file_path+param.TOY_FILE)
        elif dataset_name=='sports':
            tmp_data=extract_raw_sum(file_path+param.SPORTS_FILE)
        elif dataset_name=='movie':
            tmp_data=extract_raw_sum(file_path+param.MOVIE_FILE)
        else:
            raise ValueError('The input dataset_name \'{}\' does\'nt exist'.format(dataset_name))
        
        # on machine 13, see /data/senyang/data/sentiment/pointer_generator/decode_folder/
        if not os.path.isfile(file_path+'pointer_decode_'+dataset_name+'_train'+'.pkl'):
            decode_sum_train=load_txt_folder(decode_folder+dataset_name+'/train/')
            pickle.dump(decode_sum_train, open(file_path+'pointer_decode_'+dataset_name+'_train'+'.pkl', 'wb'), protocol=2)
        else:
            decode_sum_train = pickle.load(open(file_path+'pointer_decode_'+dataset_name+'_train'+'.pkl', 'rb'))

        decode_sum_test=load_txt_folder(decode_folder+dataset_name+'/test/')
        decode_sum_dev=load_txt_folder(decode_folder+dataset_name+'/dev/')
        tmp_data['sum']=decode_sum_dev+decode_sum_test+decode_sum_train
        # print(len(tmp_data['label']))
        print(len(decode_sum_dev), len(decode_sum_test), len(decode_sum_train))
        # print(len(tmp_data['sum']))
        ii=0
        while len(tmp_data['label'])>len(tmp_data['sum']):
            tmp_data['sum'].append(' [PAD] ')
            ii+=1
        print('padding number={}'.format(ii))

    print('\n-----Loading finished, time spent: {}-----'.format(time.time()-load_time))
        
    print('\n-----Tokenizing datset-----')
    tokenize_time=time.time()
    
    if mode=='golden':
        if os.path.isfile(file_path+'tokenized_data_'+dataset_name+'.pkl'):
            print('\n-----Tokenized data exists, load directly-----')
            tokenized_data=pickle.load(open(file_path+'tokenized_data_'+dataset_name+'.pkl','rb'))
        else:
            print('\n-----No tokenized data found, tokenizing data-----')
            tokenized_data=tokenize_data(tmp_data,tokenizer)
            pickle.dump(tokenized_data,open(file_path+'tokenized_data_'+dataset_name+'.pkl','wb'),protocol=2)
    elif mode=='predicted':
        if os.path.isfile(file_path+'tokenized_data_'+dataset_name+'_pointer.pkl'):
            print('\n-----Tokenized data exists, load directly-----')
            tokenized_data=pickle.load(open(file_path+'tokenized_data_'+dataset_name+'_pointer.pkl','rb'))
        else:
            print('\n-----No tokenized data found, tokenizing data-----')
            tokenized_data=tokenize_data(tmp_data,tokenizer)
            pickle.dump(tokenized_data,open(file_path+'tokenized_data_'+dataset_name+'_pointer.pkl','wb'),protocol=2)
    del tmp_data

    print('\n-----Tokenizing finished, time spent: {}-----'.format(time.time()-tokenize_time))
    # print(tokenized_data['sum'][2000:2050])
    
    preprocessing_time=time.time()
    if mode=='golden':
        if os.path.isfile(file_path+'word2index_'+dataset_name+'.pkl'):
            print('\n-----Processed word embedding files for \'{}\' dataset, the dataset exists and will be loaded directly-----'.format(dataset_name))
            word2index=pickle.load(open(file_path+'word2index_'+dataset_name+'.pkl','rb'))
            index2vec=pickle.load(open(file_path+'index2vec_'+dataset_name+'.pkl','rb'))
            print('\n-----Word embedding files loaded successfully, time spent: {} seconds-----'.format(time.time()-preprocessing_time))
        else:
            print('\n-----No processed word embedding files for \'{}\' dataset. So it may take some time for preprocesssing data-----'.format(dataset_name))
            word_embeddings=pickle.load(open(param.GLOVE_PATH,'rb'))
            word2index=dict()
            index2vec=[]
            # add '[PAD]' token to word_embedding
            word2index['[PAD]']=0
            index2vec.append(np.zeros(len(word_embeddings['hello'])))
            ###########################################################
            
            word2index,index2vec=construct_word_embedding(tokenized_data['raw'],word_embeddings,word2index=word2index,index2vec=index2vec)
            word2index,index2vec=construct_word_embedding(tokenized_data['sum'],word_embeddings,word2index=word2index,index2vec=index2vec)

            pickle.dump(word2index,open(file_path+'word2index_'+dataset_name+'.pkl','wb'),protocol=2)
            pickle.dump(index2vec,open(file_path+'index2vec_'+dataset_name+'.pkl','wb'),protocol=2)
    elif mode=='predicted':
        if os.path.isfile(file_path+'word2index_'+dataset_name+'_pointer.pkl'):
            print('\n-----Processed word embedding files for \'{}\' dataset, the dataset exists and will be loaded directly-----'.format(dataset_name))
            word2index=pickle.load(open(file_path+'word2index_'+dataset_name+'_pointer.pkl','rb'))
            index2vec=pickle.load(open(file_path+'index2vec_'+dataset_name+'_pointer.pkl','rb'))
            print('\n-----Word embedding files loaded successfully, time spent: {} seconds-----'.format(time.time()-preprocessing_time))
        else:
            print('\n-----No processed word embedding files for \'{}\' dataset. So it may take some time for preprocesssing data-----'.format(dataset_name))
            word_embeddings=pickle.load(open(param.GLOVE_PATH,'rb'))
            word2index=dict()
            index2vec=[]
            # add '[PAD]' token to word_embedding
            word2index['[PAD]']=0
            index2vec.append(np.zeros(len(word_embeddings['hello'])))
            ###########################################################
            
            word2index,index2vec=construct_word_embedding(tokenized_data['raw'],word_embeddings,word2index=word2index,index2vec=index2vec)
            word2index,index2vec=construct_word_embedding(tokenized_data['sum'],word_embeddings,word2index=word2index,index2vec=index2vec)

            pickle.dump(word2index,open(file_path+'word2index_'+dataset_name+'_pointer.pkl','wb'),protocol=2)
            pickle.dump(index2vec,open(file_path+'index2vec_'+dataset_name+'_pointer.pkl','wb'),protocol=2)
        print('\n-----Preprocessing finished, time spent: {} -----'.format(time.time()-preprocessing_time))
            
    weight=torch.FloatTensor(index2vec)
    embedding=nn.Embedding.from_pretrained(weight)


    labels=tokenized_data['label']
    index_sum=corpus2ids(tokenized_data['sum'],word2index)
    index_raw=corpus2ids(tokenized_data['raw'],word2index)
    if param.NOT_USE_SIMPLE==False and param.USE_SUMMARY:
        for i,_ in enumerate(index_raw):
            # index_raw[i]=index_sum[i]+index_raw[i]
            index_raw[i]+=index_sum[i]
    dev_dataset=MixedDataset(index_raw[:1000],labels[:1000],embedding,sum_ids=index_sum[:1000])
    dev_loader=tdata.DataLoader(dev_dataset,batch_size=param.BATCH_SIZE,shuffle=use_shuffle,collate_fn=batchify)
    test_dataset=MixedDataset(index_raw[1000:2000],labels[1000:2000],embedding,sum_ids=index_sum[1000:2000])
    test_loader=tdata.DataLoader(test_dataset,batch_size=param.BATCH_SIZE,shuffle=use_shuffle,collate_fn=batchify)
    train_dataset=MixedDataset(index_raw[2000:],labels[2000:],embedding,sum_ids=index_sum[2000:])
    train_loader=tdata.DataLoader(train_dataset,batch_size=param.BATCH_SIZE,shuffle=use_shuffle,collate_fn=batchify)

    if output_att_weight:
        return train_loader, dev_loader, test_loader, index_sum, index_raw, word2index
    return train_loader,dev_loader,test_loader


#===========================================================================================

'''
below are for amazon_dataset preprocessing
'''
def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield eval(l)

def load_amazon_review(file_path):
    '''
    input: 
        file_path, format: [xxx.json.gz]
    output:
        format: list of dict
        each dict contains:
            reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
            asin - ID of the product, e.g. 0000013714
            reviewerName - name of the reviewer
            helpful - helpfulness rating of the review, e.g. 2/3
            reviewText - text of the review
            overall - rating of the product
            summary - summary of the review
            unixReviewTime - time of the review (unix time)
            reviewTime - time of the review (raw)
    '''
    json_file=parse(file_path)
    data=[]
    for _ in json_file:
        data.append(_)
    return data

def extract_raw_sum(file_path):
    '''
    input: 
        file_path, format: [xxx.json.gz]
    output:
        one dict:
            label - list of labels
            raw - list of raw_text
            sum - list of summary_text
    '''
    data=load_amazon_review(file_path)
    rst=dict()
    labels=[]  #label: overall rating, list of int
    raws=[]  # raw: reviewText, list of str
    sums=[]  # sum: summary, list of str
    for _ in data:
        labels.append(int(_['overall'])-1)  # set labels from range [1,5] to [0,4]
        raws.append(_['reviewText'])
        sums.append(_['summary'])
    rst['label']=labels
    rst['raw']=raws
    rst['sum']=sums
    return rst

def construct_word_embedding(text,embedding,word2index=None,index2vec=None):
    '''
    input:
        text - list of tokenized text
        embedding - dict, from token to vector
        word2index - dict, from token to id
        index2vec - list, from index to vector
    output:
        word2index
        index2vec
    '''
    if word2index==None:
        word2index=dict()
        index2vec=[]
    for i,sent in enumerate(text):
        for j,token in enumerate(sent):
            if token not in word2index and token in embedding:
                word2index[token]=len(word2index)
                index2vec.append(embedding[token])
    return word2index,index2vec

def corpus2ids(corpus,word2idx):
    '''
    input:
        :corpus: tokenized corpus
        :word2idx: dict(),key:token,value:index
    
    output:
        index of the input corpus
    '''
    ids=[]
    tmp_ids=[]
    print('Warning (covert text to ids): The length of original sentences may be different from the length of their ids, because the UNK words are skipped.')
    for i,sent in enumerate(corpus):
        tmp_ids.clear()
        for j,token in enumerate(sent):
            try:
                tmp_ids.append(word2idx[token])
            except KeyError as e:
                continue
        ids.append(copy.deepcopy(tmp_ids))
    return ids

def tokenize_data(data,tokenizer,truncate_raw=True,truncate_sum=True):
    '''
    input:
        data - output of function 'extract_raw_sum', 
                which is one dict:
                    label - list of labels
                    raw - list of raw_text
                    sum - list of summary_text
        word_tokenizer
    output:
        one dict:
            label - list of labels
            raw - list of tokenized raw_text, in other words, list of lists of tokens
            sum - list of tokenized summary_text, in other words, list of lists of tokens
    '''
    rst_raw=[]
    rst_sum=[]
    rst=dict()
    rst['label']=data['label']
    for i,_ in enumerate(data['raw']):
        '''
        the tokenized raw_text len is restricted at 256, which can cover about 96% of all data (for 96%, it's 230; for 97%, it's 262)
        '''
        ###########     TRUNCATE here, max_len=256
        tmp_raw=tokenizer.tokenize(_)
        if truncate_raw and len(tmp_raw)>256:
            tmp_raw=tmp_raw[:256]
        rst_raw.append(copy.deepcopy(tmp_raw))
        ###################################################
        tmp_sum=tokenizer.tokenize(data['sum'][i])
        if truncate_sum and len(tmp_sum)>16:
            tmp_sum=tmp_sum[:16]
        rst_sum.append(copy.deepcopy(tmp_sum))
    rst['raw']=rst_raw
    rst['sum']=rst_sum
    return rst


def text2index(text,bert_tokenizer):
    '''
    input:
        - a list of str (str AKA: text)
        - BertTokenizer
    output:
        - a list of lists of token_index. Also, we add '[CLS]' and '[SEP]' at the beginning and the end, respectively.
           (the max length of sequence: default as 512, including [CLS] and [SEP])
    '''
    rst=[]
    for i,_ in enumerate(text):
        tmp_tokens=bert_tokenizer.tokenize(_)
        if len(tmp_tokens)>BERT_MAX_SEQ_LEN-2:
            tmp_tokens=tmp_tokens[:BERT_MAX_SEQ_LEN-2]
        tmp_tokens.insert(0,'[CLS]')
        tmp_tokens.append('[SEP]')
        rst.append(bert_tokenizer.convert_tokens_to_ids(tmp_tokens))
    return rst

def convert_text_to_index(data,bert_tokenizer,sent_tokenize=None):
    '''
    input:
        data - output of function 'extract_raw_sum', 
                which is one dict:
                    label - list of labels
                    raw - list of raw_text
                    sum - list of summary_text
        bert_tokenizer - BertTokenizer
        sent_tokenize - if set, should be similar to nltk.sent_tokenize
    output:
        - if sent_tokenize=None:
            one dict, but the text is transformed into index:
                label - list of labels
                raw - list of raw_index
                sum - list of summary_index
        - elif sent_tokenize is set:
            one dict, but the text is transformed into index:
                label - list of labels
                raw - list of sentences if raw_index
                sum - list of sentences summary_index
    '''
    rst=dict()
    rst['label']=data['label']
    if sent_tokenize is None:
        rst['raw']=text2index(data['raw'],bert_tokenizer)
        rst['sum']=text2index(data['sum'],bert_tokenizer)
    else:
        raw_rst=[]
        sum_rst=[]
        for i,_ in enumerate(data['raw']):
            raw_rst.append(text2index(sent_tokenize(_),bert_tokenizer))
            sum_rst.append(text2index(sent_tokenize(data['sum'][i]),bert_tokenizer))
        rst['raw']=raw_rst
        rst['sum']=sum_rst
    return rst