

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
import os
from itertools import islice
import operator


def init(project_list):
    global _status
    _status = {}
    for project in project_list:
        _status.update({project : Status.READY})

    global data
    data = {}
    try:
        for project in project_list:
            print(project)
            project_path = f"data/{project}/"
            print(project_path)
            file_path = os.path.join(project_path,"trainQuery.csv")
            data.update({project : pd.read_csv(file_path, encoding='utf-8')})
    except Exception:
        print("data file not found")
        exit()

    global dics_df
    dics_df = {}
    for project in project_list:
        dics_df.update({project : dict(data[project].groupby('CATEGORY')['TITLE'].apply(list))})

    global X, Y
    X = {}
    Y = {}
    for project in project_list:
        Y.update({project: copy.deepcopy(data[project]['CATEGORY'])})
        X.update({project: copy.deepcopy(data[project]['TITLE'])})

    global label_encoder
    label_encoder = LabelEncoder()



    global max_len
    max_len = {}
    try:
        for project in project_list:
            project_path = f"data/{project}/"
            file_path = os.path.join(project_path,"max_len.pickle")
            with open(file_path, 'rb') as handle:
                max_len.update({project : pickle.load(handle)})

        handle.close()
    except Exception:
        pass

    global tokenizer
    tokenizer = {}
    try:
        for project in project_list:
            project_path = f"data/{project}/"
            file_path = os.path.join(project_path, "tokenizer.pickle")
            with open(file_path, 'rb') as handle:
                tokenizer.update({project : pickle.load(handle)})

        handle.close()
    except Exception:
        pass

    global model_list
    model_list = {}
    try:
        for project in project_list:
            project_path = f"data/{project}/"
            file_path = os.path.join(project_path, "best_model.h5")
            model_list.update({project : load_model(file_path)})

    except Exception:

        pass

x = []
x.append(123)
x.append(5213)
init(x)


##형태소 분석
# input : data file(sentence, label data)
# output : morpheme set ( each sentence --> morpheme set)
# ex) 나는 집에 간다 --> 나 집 간다
def preprocess_morpheme(project_id):
    #TODO 사용자질의가 "" 일때 예외처리하기
    for i, document in enumerate(X[project_id]):
        me = Okt()

        clean_word = me.pos(document, stem=True, norm=True)
        join_word = []
        for word in clean_word:
            if not word[1] in ["Josa"]:
                join_word.append(word[0])

        document = ' '.join(join_word)
        X[project_id][i] = document


##토크나이저
# input : most_common_word ( 동의어 개수에 따라 달라질 수 있음 )
# Tokenizer
def get_tokenizer(n_most_common_words,project_id):
    _status = Status.BUSY
    tokenizer[project_id] = Tokenizer(num_words=n_most_common_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    preprocess_morpheme(project_id)
    tokenizer[project_id].fit_on_texts(X[project_id].values)
    project_path = f"data/{project_id}/"
    file_path = os.path.join(project_path, "tokenizer.pickle")
    with open(file_path, 'wb') as handle:
        pickle.dump(tokenizer[project_id], handle)
    handle.close()
    return tokenizer[project_id]


##encodeing + padding
def create_train_vector(project_id):
    sequences = tokenizer[project_id].texts_to_sequences(X[project_id].values)
    max_len[project_id] = max(len(l) for l in sequences)
    project_path = f"data/{project_id}/"
    file_path = os.path.join(project_path, "max_len.pickle")
    with open(file_path, 'wb') as handle:
        pickle.dump(max_len[project_id], handle)
    handle.close()
    _pad_sequences = pad_sequences(sequences, max_len[project_id])
    return _pad_sequences


def train_label_vector(project_id):
    encoded_label = label_encoder.fit_transform(Y[project_id].tolist())

    y_train_one = to_categorical(encoded_label)
    print(y_train_one.shape)
    return y_train_one


def jaro(sentence, query):
    score = _levenshtein.ratio(sentence, query)
    return score


def get_key(val,project_id):
    for key, value in dics_df[project_id].items():

        if val in value:
            return key

    return -1


def similarity(query, threshold,project_id):
    query_dict = {}
    for v in data[project_id]['TITLE']:

        score = jaro(v, query)

        if score >= threshold:
            # map에 집어넣기
            query_dict.update({v: score})

            # map에서 가장 높은 놈 topk 만큼 뽑기
    result = sorted(query_dict.items(), reverse=True, key=operator.itemgetter(1))[:5]

    return result

def similarity_top_k(query, threshold, top_k,project_id):
    query_dict = []

    count = 0
    for i, document in enumerate(data[project_id]['TITLE']):
        count = count + 1
        score = jaro(document, query)

        if score >= threshold:
            sim_data = data_object.similarity_data(data[project_id]['CATEGORY'][i], document, score)

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


def cnn_model(dense_num, maxlen,project_id):
    model = Sequential()
    word_index = tokenizer[project_id].word_index

    model.add(Embedding(len(word_index) + 1, 128, input_length=maxlen))
    model.add(Conv1D(filters=512, kernel_size=2, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Flatten())

    model.add(Dense(dense_num, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model.summary()
    return model


def train_model(model_select, sentences, labels,project_id):
    es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=4)
    mc = ModelCheckpoint(f'data/{project_id}/best_model.h5', monitor='acc', mode='max', verbose=1, save_best_only=True)
    trained_model = model_select.fit(sentences, labels, batch_size=128, epochs=15, callbacks=[es, mc])

    return trained_model


# dicts_df = (dict(data.groupby('CATEGORY')['TITLE'].apply(list)))
# dicts = copy.deepcopy

def train_status(project_id):
    print(_status)
    if _status[project_id] == Status.READY:
        return True
    else:
        return False


def train(project_id):
    global _status

    _status.update({project_id: Status.BUSY})

    t = threading.Thread(target=retrain, args=(project_id,))
    t.start()


def retrain(project_id):

    global _status

    _status[project_id] = Status.BUSY

    tokenizer[project_id] = get_tokenizer(5000,project_id)
    pad_X = create_train_vector(project_id)
    y_train_one = train_label_vector(project_id)
    model = train_model(cnn_model(len(data[project_id]['CATEGORY'].unique()), max_len[project_id],project_id), pad_X, y_train_one,project_id)
    _status[project_id] = Status.READY


# loaded_model = load_model('model.h5')

# def analyze_query(sentence):
#


def sentence_classification(sentence,project_id):
    check_time = time.time()
    sentence_morpheme = morphs(sentence)
    seq = tokenizer[project_id].texts_to_sequences([sentence_morpheme])
    train_label_vector(project_id)
    padded = pad_sequences(seq, maxlen=max_len[project_id])
    preds = model_list[project_id].predict(padded)
    print(time.time() - check_time)
    print("-----------------------------------predict end-----------------")
    pandass = pd.Series(preds[0],Y[project_id].unique())
    pandass = pandass.sort_values(ascending=False)

    print("-----------------------------------pandass end-----------------")
    result = []
    for index, item in zip(range(3), pandass.items()):
        print(item)
        result.append((list(item)))
    print("------------------------result end ----------------------------")


    return result


def sentence_similarity(sentence, threshold,project_id):
    set = similarity(sentence, threshold,project_id)

    if not set:
        return -1

    return get_key(set[0][0],project_id)




def morphs(sentence):
    me = Okt()
    clean_word = me.pos(sentence, stem=True, norm=True)
    join_word = []
    for word in clean_word:
        if not word[1] in ["Josa"]:
            join_word.append(word[0])

    document = ' '.join(join_word)

    return document


# retrain(123)
# retrain(5213)
#
# print(sentence_classification("온비드는 뭐냐",123))
# print(sentence_similarity("온비드에서 왜 일반공인인증서는 안됩니까?",0.55,123))
# print(sentence_similarity("택배 신규 예약해줘",0.55,5213))
print("Hello")
# train()