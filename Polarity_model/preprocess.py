import difflib
import itertools
import numpy as np
import re
import enum
import collections
import random
import copy
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer

def clean_text(text, only_upper=False):
    # should there be a str here?`
    text = '%s%s' % (text[0].upper(), text[1:])
    if only_upper:
        return text
    text = text.replace('|', 'UNK')
    text = re.sub('(^|\s)-($|\s)', r'\1@-@\2', text)
    # text = re.sub(' (n?\'.) ', r'\1 ', text)
    # fix apostrophe stuff according to tokenizer
    text = re.sub(' (n)(\'.) ', r'\1 \2 ', text)
    return text

def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    if n > flat.shape[0]:
        indices = np.array(range(flat.shape[0]), dtype='int')
        return np.unravel_index(indices, ary.shape)
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)

class OpToken:
    def __init__(self, type_, value):
        self.type = type_
        self.value = value

    def test(self, token):
        if self.type == 'text':
            return token.text == self.value
        if self.type == 'pos':
            return token.pos == self.value
        if self.type == 'tag':
            return token.tag == self.value

    def hash(self):
        return self.type + '_' + self.value

Token = collections.namedtuple('Token', ['text', 'pos', 'tag'])

def capitalize(text):
    if len(text) == 0:
        return text
    if len(text) == 1:
        return text.upper()
    else:
        return '%s%s' % (text[0].upper(), text[1:])


class TextProcessor:
    '''
        Performs text preprocessing for input data. 
        1. Adds a special [CLS] token for transformer classifier
        2. Calculates the max sequence length required for the model
        3. Returns padded seuences with word index from the tokenizer
        
        fit
            fits tokenizer on the data and calculate max_sequence_length
        fit_and_transform
            fits on the given data and returns the padded sequnces
        transform
            transform the given test data
    '''
    def __init__(self, nlp):
        self.nlp = nlp
        self.tokenizer= Tokenizer()
        self.max_seq_len=None

    def fit_and_transform(self,texts):
        texts = self.clean_for_model(texts)
        texts = self.addSpecialToken(texts)

        self.tokenizer.fit_on_texts(texts)
        self.max_seq_len = max([len(text.split()) for text in texts])

        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences,padding='post',maxlen=self.max_seq_len)

        return padded_sequences

    def fit(self,texts):
        texts = self.clean_for_model(texts)
        texts = self.addSpecialToken(texts)

        self.max_seq_len = max([len(text.split()) for text in texts])
        self.tokenizer.fit_on_texts(texts)
        
        return None

    def transform(self,texts):
        texts = self.clean_for_model(texts)
        texts = self.addSpecialToken(texts)

        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences,padding='post',maxlen = self.max_seq_len)

        return padded_sequences


    def tokenize(self, texts):
        ret = []
        processed = self.nlp.pipe(texts)
        for text, p in zip(texts, processed):
            token_sequence = [Token(x.text, x.pos_, x.tag_) for x in p]
            ret.append(token_sequence)
        return ret
    def tokenize_text(self, texts):
        return [' '.join([a.text for a in x]).lower() for x in self.nlp.tokenizer.pipe(texts)]

    def clean_for_model(self, texts):
        fn = lambda x: re.sub(r'\s+', ' ', re.sub(r'\s\'(\w{1, 3})', r"'\1", x).replace('@-@', '-').strip())
        return self.tokenize_text([fn(capitalize(x)) for x in texts])

    def clean_for_humans(self, texts):
        return [re.sub("\s(n')", r'\1', re.sub(r'\s\'(\w)', r"'\1", capitalize(x))) for x in texts]

    def addSpecialToken(self,quesition_list):
        return ['[CLS] ' + ques for ques in quesition_list]
