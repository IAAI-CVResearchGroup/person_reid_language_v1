from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
from PIL import Image
import numpy as np
import os.path as osp
import io
import pickle
import torch
from torch.utils.data import Dataset
#import gensim
#from gensim.test.utils import common_texts, get_tmpfile
#from gensim.models import Word2Vec
#import nltk
import json
import random
import pdb
import re

#word_model = gensim.models.KeyedVectors.load_word2vec_format('/home/anqing-test/xj/data/GoogleNews-vectors-negative300.bin.gz', binary=True)


root_path = '/data/jun.xu/'

with open(root_path + 'CUHK-PEDES/caption_all.json') as fin:
    data = json.load(fin)

all_captions = {}
for caption in data:
    all_captions[caption['file_path']] = caption['captions']

def read_sentence(img_path):
    imgs = img_path.split('/')
    img_name = imgs[-1]
    #print(imgs)
    ID = int(img_name[2:5])
    #print(ID)
    num = random.randint(1, 4)
    key = str(ID)+'_'+str(num)+'.jpg'    
    key_image = 'CUHK03/'+ key
    path = root_path + 'CUHK-PEDES/imgs/'+key_image
    isfile = os.path.exists(path)
    while not(isfile):
        num = random.randint(1, 4)
        key = str(ID)+'_'+str(num)+'.jpg'
        key_image = 'CUHK03/'+ key
        path = root_path + 'CUHK-PEDES/imgs/'+key_image
        isfile = os.path.exists(path)
        #if isfile == True:
           # break
    sentence = all_captions[key_image]
    num_captions=random.randint(0,1)
    single_sentence = sentence[num_captions]
    #print(single_sentence)
    list_sent = single_sentence.strip().split()
    #print(list_sent)
    worddict_tmp = pickle.load(open('data/wordlist_reid.p', 'rb'))
    wordlist = [l for l in iter(worddict_tmp.keys()) if l != '</S>']
    wordlist_final = ['EOS'] + sorted(wordlist)
    word_vectors_all = torch.LongTensor(30).zero_()
    punc = '[,.]'
    #caption_gen=''
    for i, word in enumerate(list_sent):
        if i >= 30:
            break

        word = re.sub(punc, '', word)
        word = word.lower()

        if (word not in wordlist_final):
            word = 'UNK'
        word_vectors_all[i] = wordlist_final.index(word)
        #caption_gen +=' '+word

    caption_raw = single_sentence
    #print(caption_raw)
    #print(caption_gen)
    return word_vectors_all, caption_raw
   
def read_image(img_path):
    #print(img_path)
    #imgs = img_path.split('/')
    #print(len(imgs))
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)
        caption, caption_raw = read_sentence(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid, caption, caption_raw


class VideoDataset(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, seq_len=15, sample='evenly', transform=None):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)

        if self.sample == 'random':
            """
            Randomly sample seq_len items from num items,
            if num is smaller than seq_len, then replicate items
            """
            indices = np.arange(num)
            replace = False if num >= self.seq_len else True
            indices = np.random.choice(indices, size=self.seq_len, replace=replace)
            # sort indices to keep temporal order (comment it to be order-agnostic)
            indices = np.sort(indices)
        elif self.sample == 'evenly':
            """
            Evenly sample seq_len items from num items.
            """
            if num >= self.seq_len:
                num -= num % self.seq_len
                indices = np.arange(0, num, num/self.seq_len)
            else:
                # if num is smaller than seq_len, simply replicate the last image
                # until the seq_len requirement is satisfied
                indices = np.arange(0, num)
                num_pads = self.seq_len - num
                indices = np.concatenate([indices, np.ones(num_pads).astype(np.int32)*(num-1)])
            assert len(indices) == self.seq_len
        elif self.sample == 'all':
            """
            Sample all items, seq_len is useless now and batch_size needs
            to be set to 1.
            """
            indices = np.arange(num)
        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))

        imgs = []
        for index in indices:
            img_path = img_paths[int(index)]
            img = read_image(img_path)
            if self.transform is not None:
                img = self.transform(img)
            img = img.unsqueeze(0)
            imgs.append(img)
        imgs = torch.cat(imgs, dim=0)

        return imgs, pid, camid
