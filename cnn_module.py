import numpy as np  # linear algebra
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
from konlpy.tag import Okt
from Levenshtein import _levenshtein
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Flatten
from tensorflow.keras.preprocessing.sequence import pad_sequences
from status_train import Status
import pickle
import data_object
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
import threading
import copy
import operator


def init():
    global _status
    _status = Status.READY

    global data
    data = pd.read_csv('data/koreanjasan.csv', encoding='utf-8')

    global dics_df
    dics_df = dict(data.groupby('CATEGORY')['TITLE'].apply(list))

    global X, Y
    Y = copy.deepcopy(data['CATEGORY'])
    X = copy.deepcopy(data['TITLE'])

    global label_encoder
    label_encoder = LabelEncoder()


    global max_len
    try:
        with open('data/max_len.pickle', 'rb') as handle:
            max_len = pickle.load(handle)
        handle.close()
    except Exception:
        pass

    global tokenizer
    try:
        with open('data/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        handle.close()
    except Exception:
        pass

    global model, model_list
    try:
        file = 'best_model.h5'

        model = load_model(file)

    except Exception:

        pass


init()


##형태소 분석
# input : data file(sentence, label data)
# output : morpheme set ( each sentence --> morpheme set)
# ex) 나는 집에 간다 --> 나 집 간다
def preprocess_morpheme():
    #TODO 사용자질의가 "" 일때 예외처리하기
    for i, document in enumerate(X):
        me = Okt()
        print(i,document)
        clean_word = me.pos(document, stem=True, norm=True)
        join_word = []
        for word in clean_word:
            if not word[1] in ["Josa"]:
                join_word.append(word[0])

        document = ' '.join(join_word)
        X[i] = document


##토크나이저
# input : most_common_word ( 동의어 개수에 따라 달라질 수 있음 )
# Tokenizer
def get_tokenizer(n_most_common_words):
    _status = Status.BUSY
    tokenizer = Tokenizer(num_words=n_most_common_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    preprocess_morpheme()
    tokenizer.fit_on_texts(X.values)
    with open('data/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle)
    handle.close()
    return tokenizer


##encodeing + padding
def create_train_vector():
    sequences = tokenizer.texts_to_sequences(X.values)
    max_len = max(len(l) for l in sequences)
    with open('data/max_len.pickle', 'wb') as handle:
        pickle.dump(max_len, handle)
    handle.close()
    _pad_sequences = pad_sequences(sequences, max_len)
    return _pad_sequences


def train_label_vector():
    encoded_label = label_encoder.fit_transform(Y.tolist())

    y_train_one = to_categorical(encoded_label)
    print(y_train_one.shape)
    return y_train_one


def jaro(sentence, query):
    score = _levenshtein.ratio(sentence, query)
    return score


def get_key(val):
    for key, value in dics_df.items():

        if val in value:
            return key

    return -1


def similarity(query, threshold):
    query_dict = {}
    for v in data['TITLE']:

        score = jaro(v, query)

        if score >= threshold:
            # map에 집어넣기
            query_dict.update({v: score})

            # map에서 가장 높은 놈 topk 만큼 뽑기
    result = sorted(query_dict.items(), reverse=True, key=operator.itemgetter(1))[:5]

    return result

def similarity_top_k(query, threshold, top_k):
    query_dict = []

    count = 0
    for i, document in enumerate(data['TITLE']):
        count = count + 1
        score = jaro(document, query)

        if score >= threshold:
            sim_data = data_object.similarity_data(data['CATEGORY'][i], document, score)

            query_dict.append(sim_data)
    query_dict.sort(key=operator.attrgetter('score'),reverse=True)
    sorted(query_dict,key=lambda d: int(d.id), reverse=True)



    result =[]
    id_list = []
    for v in query_dict:
        if str(v.id) not in id_list:
            result.append(v)
            id_list.append(str(v.id))

    return result[:top_k]




# dense_num = category_num
# max_len = 한 문장 최대 길이
def lstm_model(word_index, dense_num):
    model = Sequential()
    model.add(Embedding(len(word_index) + 1, 100, input_length=max_len))
    model.add(LSTM(max_len))
    model.add(Dense(dense_num, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return model


def cnn_model(dense_num, maxlen):
    model = Sequential()
    word_index = tokenizer.word_index

    model.add(Embedding(len(word_index) + 1, 128, input_length=maxlen))
    model.add(Conv1D(filters=512, kernel_size=2, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Flatten())

    model.add(Dense(dense_num, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model.summary()
    return model


def train_model(model_select, sentences, labels):
    es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=4)
    mc = ModelCheckpoint('best_model.h5', monitor='acc', mode='max', verbose=1, save_best_only=True)
    trained_model = model_select.fit(sentences, labels, batch_size=128, epochs=20, callbacks=[es, mc])

    return trained_model


# dicts_df = (dict(data.groupby('CATEGORY')['TITLE'].apply(list)))
# dicts = copy.deepcopy

def train_status():

    if _status == Status.READY:
        return True
    else:
        return False


def train():
    global _status
    _status = Status.BUSY

    t = threading.Thread(target=retrain, args=())
    t.start()


def retrain():

    global _status
    _status = Status.BUSY

    tokenizer = get_tokenizer(5000)
    pad_X = create_train_vector()
    y_train_one = train_label_vector()
    train_model(cnn_model(len(data['CATEGORY'].unique()), max_len), pad_X, y_train_one)
    _status = Status.READY

# loaded_model = load_model('model.h5')

# def analyze_query(sentence):
#


def sentence_classification(sentence):
    sentence_morpheme = morphs(sentence)
    seq = tokenizer.texts_to_sequences([sentence_morpheme])
    train_label_vector()
    padded = pad_sequences(seq, maxlen=max_len)
    preds = model.predict(padded)

    index = np.argmax(preds)


    label = label_encoder.classes_[index]

    result = {sentence: label}
    return result


def sentence_similarity(sentence, threshold):
    set = similarity(sentence, threshold)

    if not set:
        return -1

    return get_key(set[0][0])




def morphs(sentence):
    me = Okt()
    clean_word = me.pos(sentence, stem=True, norm=True)
    join_word = []
    for word in clean_word:
        if not word[1] in ["Josa"]:
            join_word.append(word[0])

    document = ' '.join(join_word)

    return document

# retrain()

print(sentence_classification("온비드는 뭐냐"))
print(sentence_similarity("온비드에서 왜 일반공인인증서는 안됩니까?",0.55))
# print("Hello")
# train()