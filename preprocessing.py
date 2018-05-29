import numpy as np
from collections import Counter
import itertools
from konlpy.tag import *;t = Twitter();
def data_and_labels():


    class0 = list(open('k0',encoding='utf-8').readlines())
    class0 = [s.strip() for s in class0]


    class3 = list(open('k4', encoding='utf-8').readlines())
    class3 = [s.strip() for s in class3]

    x_text = class0+class3
    x_text = [s.split(" ") for s in x_text]

    class0_label = [0 for _ in class0]
    class3_label = [1 for _ in class3]


    y = np.concatenate([class0_label,class3_label], 0)

    return [x_text,y]


def build_vocab(sentences):
    word_counts = Counter(itertools.chain(*sentences))
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]

def pad_sentences(sentences, padding_word="<PAD/>"):

    pos = lambda d: [ '/'.join(t) for t in t.pos(d)]
    pos_sentences = []
    for i in sentences:
        i = pos(' '.join(i))
        pos_sentences.append(i)
    sequence_length = max(len(x) for x in pos_sentences)
    padded_sentences = []
    print(sequence_length)
    for i in range(len(pos_sentences)):
        sentence = pos_sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences

def load_data():
    sentences, labels = data_and_labels()
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


