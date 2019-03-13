# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from static import *
import torch.utils.data as data
from gensim.models import KeyedVectors

# file_debug = open("debug.txt", "w")
# vocab_one_hot = open(VOCAB_ONE_HOT, "w")
# print(UNKNOWN, 0, file=vocab_one_hot)
# print(NUMBER, 1, file=vocab_one_hot)
# vocab_unk = 0
# vocab = [UNKNOWN, NUMBER]


class SC_DATA(data.Dataset):
    DEBUG = False

    # inputs    : Batch x Length x Word2Vec
    # outputs   : Batch x Length x Tag2Vec
    # masks     : Batch x Length x Mask
    # list_chars: Batch x Length x Length_word x Char2Vec
    # mask_chars: Batch x Length x Length_word x Mask_len

    def get_line(self, index):
        sentence, labels = self.list_sentence[index], self.list_labels[index]
        # list_words, len_1 = self.sentence_to_list_word(sentence)
        list_words = self.list_sentence_vec[index]
        list_label, len_2 = self.labels_to_list_label(labels)
        stmc = self.sentence_to_matrix_char(sentence)
        le = len_2
        lle = []
        for i in range(MAX_LENGTH_SENTENCE):
            if i < le:
                lle.append(np.array([1]))
            else:
                lle.append(np.array([0]))
        list_words_ans = list_words
        out_0 = np.array(list_words_ans)
        out_1 = np.array(list_label)
        out_2 = np.array(lle)
        out_3 = np.array(stmc[0])
        out_4 = np.array(stmc[1])

        return out_0, out_1, out_2, out_3, out_4

    def __init__(self, type_data, transform=None, target_transform=None):
        super(SC_DATA, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.maxFold = 5
        self.embed_word2vec = KeyedVectors.load_word2vec_format(FILE_WORD_2_VEC, limit=500000)
        self.embed_tag2int = KeyedVectors.load_word2vec_format(FILE_TAG_2_INT)
        self.embed_char2vec = KeyedVectors.load_word2vec_format(FILE_CHAR_2_VEC)
        self.len = 0
        self.list_file = []
        path = RESOURCES_DATA_TEST
        if type_data == "train":
            path = RESOURCES_DATA_TRAIN
        sentence = ""
        labels = ""
        self.list_sentence = []
        self.list_labels = []
        self.list_sentence_vec = []
        max_len = 0
        max_len_w = 0
        for i in range(1):
            path_file = path + type_data + str(i + 1) + ".txt"
            self.list_file.append(path_file)
            file_data = open(path_file)
            for line in file_data:
                if line == "\n":
                    max_len = max(max_len, len(sentence.split(" ")))
                    if len(sentence.split(" ")) <= 100:
                        sentence = sentence[:-1]
                        labels = labels[:-1]
                        self.list_sentence.append(sentence)
                        self.list_sentence_vec.append(self.sentence_to_list_word(sentence)[0])
                        self.list_labels.append(labels)
                        self.len += 1
                    # if self.len == mm:
                    #    return
                    sentence = ""
                    labels = ""
                else:
                    word = line.split(" ")[0] + " "
                    max_len_w = max(max_len_w, len(word))
                    sentence = sentence + line.split(" ")[0] + " "
                    labels = labels + line.split(" ")[11].replace("\n", "") + " "
        print(max_len)
        print(max_len_w)
        del self.embed_word2vec
        self.embed_word2vec = None

    def sentence_to_list_word(self, sentence):
        list_word = []
        for word in sentence.split(' '):
            if word == "\n":
                continue
            word = word.decode('utf-8')
            if self.DEBUG:
                if word not in vocab:
                    print(word.encode('utf-8'), len(vocab), file=vocab_one_hot)
                    vocab.append(word)
            if word not in self.embed_word2vec.wv.vocab:
                ok = False
                for c in word:
                    if c.isdigit():
                        word = NUMBER
                        ok = True
                        break
                if not ok:
                    if self.DEBUG:
                        print('bug', word.encode('utf-8'), "'", sentence, "'", file=file_debug)
                    word = UNKNOWN
            list_word.append(self.embed_word2vec.wv[word])
        length = len(list_word)
        while len(list_word) < MAX_LENGTH_SENTENCE:
            list_word.append(self.embed_word2vec.wv[UNKNOWN])
        return list_word, length

    def labels_to_list_label(self, labels):
        list_label = []
        for word in labels.split(' '):
            if word == "\n" or word == "":
                continue
            list_label.append(int(self.embed_tag2int.wv[word][0]))
        length = len(list_label)
        while len(list_label) < MAX_LENGTH_SENTENCE:
            list_label.append(0)
        return list_label, length

    def word_to_list_char(self, word):
        list_char = []
        for c in word:
            if c in self.embed_char2vec.wv.vocab:
                list_char.append(np.array(self.embed_char2vec.wv[c]))
            else:
                list_char.append(np.array(self.embed_char2vec.wv["'"]))
        length = len(list_char)
        while len(list_char) < MAX_LENGTH_WORD:
            list_char.append(np.array(self.embed_char2vec.wv["'".decode('utf-8')]))
        return list_char, length

    def sentence_to_matrix_char(self, sentence):
        list_char_vec = []  # list_char_vec
        mask_chars = []
        list_word = []
        for word in sentence.split(' '):
            word = word.decode('utf-8')
            if word == "\n":
                continue
            list_word.append(word)
        while len(list_word) < MAX_LENGTH_SENTENCE:
            list_word.append(UNKNOWN)

        for word in list_word:
            list_char, length = self.word_to_list_char(word)
            list_char_vec.append(np.array(list_char))
            mask = []
            for i in range(MAX_LENGTH_WORD):
                if i == length - 1:
                    mask.append(np.array([1]))
                else:
                    mask.append(np.array([0]))
            mask_chars.append(mask)
        return list_char_vec, mask_chars

    def __getitem__(self, index):
        return self.get_line(index)

    def __len__(self):
        return self.len


# from tqdm import tqdm
#
# a = SC_DATA("train")
# lenn = a.__len__()
# for i in tqdm(range(lenn)):
#    a.__getitem__(i)
#
# a = SC_DATA("test")
# lenn = a.__len__()
# for i in tqdm(range(lenn)):
#    a.__getitem__(i)
#
# print(vocab_unk)
# print(len(vocab))
