import errno

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
from pandas.io.json import json_normalize
import threading
import copy
import os
import shutil
import operator
import urllib
import json
import requests


def create_project_df_simulation(project_id):
    response = requests.get('http://127.0.0.1:17405/knowledge/get?type=triggeringSentence&projectId=' + str(project_id))
    response.encoding = 'utf-8'

    json_data = json.loads(response.text)
    df = json_normalize(json_data['items'])
    file_path = f'data/{project_id}/simulation/'
    file = os.path.join(file_path, str(project_id) + ".json")
    with open(file, 'w', encoding='utf8') as outfile:
        outfile.write(json.dumps(json_data, ensure_ascii=False))

    outfile.close()
    print(df)
    return df


def create_project_df_service(project_id):
    file_path = f'data/{project_id}/service/{project_id}.json'
    with open(file_path, encoding='utf8') as f:
        data_df = json.loads(f.read())

    df = json_normalize(data_df['items'])
    return df


def apply_model(project_id, symlinks=False, ignore=None):
    src = f'data/{project_id}/simulation'
    dst = f'data/{project_id}/service'
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)
    pjo_list = []
    pjo_list.append(project_id)
    load_service(pjo_list)


def load_service(project_list):
    ##디렉토리 없으면 디렉토리 생성
    try:
        for project in project_list:
            data_path = f"data/{project}"

            service_project_path = os.path.join(data_path, "service")
            service_project_path = str(service_project_path)

            if not (os.path.isdir(service_project_path)):
                os.makedirs(os.path.join(service_project_path))

    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory!!!!!")
            raise

    # 데이터 불러오기
    global service_data
    service_data = {}
    try:
        for project in project_list:
            service_data.update(({project: create_project_df_service(project)}))
    except Exception:
        print("service data file not found", project)
        pass

    # 데이터 dictionary로 사용(sim_sentence 사용)
    global service_dict_df
    service_dict_df = {}
    for project in project_list:
        if project in service_data:
            service_dict_df.update(
                {project: dict(service_data[project].groupby('dialogTaskId')['sentence'].apply(list))})

    # 형태소 분석 안된 pure 데이터
    global X_service, Y_service

    X_service = {}
    Y_service = {}
    for project in project_list:
        if project in service_data:
            Y_service.update({project: copy.deepcopy(service_data[project]['dialogTaskId'])})
            X_service.update({project: copy.deepcopy(service_data[project]['sentence'])})

    # 인텐트 id 인코더
    global label_encoder
    label_encoder = LabelEncoder()

    # 토크나이저
    global service_tokenizer
    service_tokenizer = {}
    try:
        for project in project_list:
            service_project_path = f"data/{project}/service"
            file_path = os.path.join(service_project_path, "tokenizer.pickle")
            with open(file_path, 'rb') as handle:
                service_tokenizer.update({project: pickle.load(handle)})

        handle.close()
    except Exception:
        pass

    # 서비스 모델
    global service_model_list
    service_model_list = {}
    try:
        for project in project_list:
            service_project_path = f"data/{project}/service"
            file_path = os.path.join(service_project_path, "best_model.h5")
            model_list.update({project: load_model(file_path)})

    except Exception:
        print("model upload failure - ", project)
        pass


def load_simulation(project_list):
    ##디렉토리 없으면 디렉토리 생성
    try:
        for project in project_list:
            data_path = f"data/{project}"

            simulation_project_path = os.path.join(data_path, "simulation")
            simulation_project_path = str(simulation_project_path)

            if not (os.path.isdir(simulation_project_path)):
                os.makedirs(os.path.join(simulation_project_path))

    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory!!!!!")
            raise

    # 데이터 불러오기
    global _status
    _status = {}
    for project in project_list:
        _status.update({project: Status.READY})

    global data
    data = {}
    try:
        for project in project_list:
            data.update({project: create_project_df_simulation(project)})
    except Exception:
        print("simulation data file not found -  ", project)
        pass

    # 데이터 dictionary로 사용(sim_sentence 사용)
    global dict_df
    dict_df = {}
    for project in project_list:
        if project in data:
            dict_df.update({project: dict(data[project].groupby('dialogTaskId')['sentence'].apply(list))})

    # 형태소 분석 안된 pure 데이터
    global X, Y
    X = {}
    Y = {}
    for project in project_list:
        if project in data:
            Y.update({project: copy.deepcopy(data[project]['dialogTaskId'])})
            X.update({project: copy.deepcopy(data[project]['sentence'])})

    # 인텐트 id 인코더
    global label_encoder
    label_encoder = LabelEncoder()

    global max_len
    max_len = {}
    try:
        for project in project_list:
            project_path = f"data/simulation/{project}/"
            file_path = os.path.join(project_path, "max_len.pickle")
            with open(file_path, 'rb') as handle:
                max_len.update({project: pickle.load(handle)})

        handle.close()
    except Exception:
        pass

    # 토크나이저
    global tokenizer
    tokenizer = {}
    try:
        for project in project_list:
            project_path = f"data/{project}/simulation/"
            file_path = os.path.join(project_path, "tokenizer.pickle")
            with open(file_path, 'rb') as handle:
                tokenizer.update({project: pickle.load(handle)})

        handle.close()
    except Exception:
        pass

    global model_list
    model_list = {}
    try:
        for project in project_list:
            project_path = f"data/simulation{project}/"
            file_path = os.path.join(project_path, "best_model.h5")
            model_list.update({project: load_model(file_path)})

    except Exception:
        pass


def delete_project(project_id):
    data_path = "data/"
    project_path = os.path.join(data_path, str(project_id))
    backup_path = "backup/"

    backup_path = str(backup_path)
    backup_pj_path = os.path.join(backup_path, str(project_id))
    if os.path.isdir(backup_pj_path):
        shutil.rmtree(backup_pj_path)

    print(model_list)
    print(service_model_list)
    id = -1
    if project_id in service_model_list.keys():
        id = service_model_list.pop(project_id)
    if project_id in model_list.keys():
        id = model_list.pop(project_id)

    if id == -1:
        print("not exists")
        return False
    shutil.move(project_path, backup_path)
    print("Delete Project : ", project_id)
    return True


##형태소 분석
# input : data file(sentence, label data)
# output : morpheme set ( each sentence --> morpheme set)
# ex) 나는 집에 간다 --> 나 집 간다
def preprocess_morpheme(project_id):
    # TODO 사용자질의가 "" 일때 예외처리하기
    print("data : ", data
          )
    print("X : ", X)
    for i, document in enumerate(X.get(project_id)):
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
def get_tokenizer(n_most_common_words, project_id):
    _status[project_id] = Status.BUSY
    tokenizer[project_id] = Tokenizer(num_words=n_most_common_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~',
                                      lower=True)
    preprocess_morpheme(project_id)
    tokenizer[project_id].fit_on_texts(X[project_id].values)
    project_path = f"data/{project_id}/simulation/"
    file_path = os.path.join(project_path, "tokenizer.pickle")
    with open(file_path, 'wb') as handle:
        pickle.dump(tokenizer[project_id], handle)
    handle.close()
    return tokenizer[project_id]


##encodeing + padding
def create_train_vector(project_id):
    sequences = tokenizer[project_id].texts_to_sequences(X[project_id].values)
    max_len[project_id] = max(len(l) for l in sequences)
    project_path = f"data/{project_id}/simulation/"
    file_path = os.path.join(project_path, "max_len.pickle")
    with open(file_path, 'wb') as handle:
        pickle.dump(max_len[project_id], handle)
    handle.close()
    _pad_sequences = pad_sequences(sequences, max_len[project_id])
    return _pad_sequences


def train_label_vector(project_id, mode):
    if mode == 'simulation':
        encoded_label = label_encoder.fit_transform(Y[project_id].tolist())
        y_train_one = to_categorical(encoded_label)
        return y_train_one
    else:
        encoded_label = label_encoder.fit_transform(Y_service[project_id].tolist())
        y_train_one = to_categorical(encoded_label)
        return y_train_one


def jaro(sentence, query):
    score = _levenshtein.ratio(sentence, query)
    return score


def get_key(val, project_id):
    for key, value in dict_df[project_id].items():

        if val in value:
            return key

    return -1


def get_key_service(val, project_id):
    for key, value in dict_df[project_id].items():

        if val in value:
            return key

    return -1


# working / service 분리 필요
# def similarity(query, threshold,project_id):
#     query_dict = {}
#     for v in data[project_id]['sentence']:
#
#         score = jaro(v, query)
#
#         if score >= threshold:
#             # map에 집어넣기
#             query_dict.update({v: score})
#
#             # map에서 가장 높은 놈 topk 만큼 뽑기
#     result = sorted(query_dict.items(), reverse=True, key=operator.itemgetter(1))[:5]
#
#     return result

# working / service 분리 필요
def similarity_top_k(query, threshold, top_k, project_id):
    query_dict = []

    count = 0
    for i, document in enumerate(data[project_id]['sentence']):
        count = count + 1
        score = jaro(document, query)

        if score >= threshold:
            sim_data = data_object.similarity_data(data[project_id]['dialogTaskId'][i], document, score)

            query_dict.append(sim_data)
    query_dict.sort(key=operator.attrgetter('score'), reverse=True)
    sorted(query_dict, key=lambda d: int(d.id), reverse=True)

    result = []
    id_list = []
    for v in query_dict:
        if str(v.id) not in id_list:
            result.append(v)
            id_list.append(str(v.id))

    return result[:top_k]


def similarity_top_k_service(query, threshold, top_k, project_id):
    query_dict = []

    count = 0
    for i, document in enumerate(service_data[project_id]['sentence']):
        count = count + 1
        score = jaro(document, query)

        if score >= threshold:
            sim_data = data_object.similarity_data(service_data[project_id]['dialogTaskId'][i], document, score)

            query_dict.append(sim_data)
    query_dict.sort(key=operator.attrgetter('score'), reverse=True)
    sorted(query_dict, key=lambda d: int(d.id), reverse=True)

    result = []
    id_list = []
    for v in query_dict:
        if str(v.id) not in id_list:
            result.append(v)
            id_list.append(str(v.id))

    return result[:top_k]


# dense_num = dialogTaskId_num
# max_len = 한 문장 최대 길이
def lstm_model(word_index, dense_num):
    model = Sequential()
    model.add(Embedding(len(word_index) + 1, 100, input_length=max_len))
    model.add(LSTM(max_len))
    model.add(Dense(dense_num, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return model


def cnn_model(dense_num, maxlen, project_id):
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


def train_model(model_select, sentences, labels, project_id):
    es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=4)
    mc = ModelCheckpoint(f'data/{project_id}/simulation/best_model.h5', monitor='acc', mode='max', verbose=1,
                         save_best_only=True)
    trained_model = model_select.fit(sentences, labels, batch_size=128, epochs=15, callbacks=[es, mc])

    return trained_model


# dicts_df = (dict(data.groupby('dialogTaskId')['sentence'].apply(list)))
# dicts = copy.deepcopy

def train_status(project_id):
    print(_status)
    if _status.get(project_id) == Status.READY:
        return True
    else:
        return False


def train(project_id):
    _status.update({project_id: Status.BUSY})

    t = threading.Thread(target=retrain, args=(project_id,))
    t.start()


def retrain(project_id):
    global _status

    _status[project_id] = Status.BUSY
    data.update({project_id: create_project_df_simulation(project_id)})
    tokenizer[project_id] = get_tokenizer(5000, project_id)
    pad_X = create_train_vector(project_id)
    y_train_one = train_label_vector(project_id, "simulation")
    model = train_model(cnn_model(len(data[project_id]['dialogTaskId'].unique()), max_len[project_id], project_id),
                        pad_X, y_train_one, project_id)
    _status[project_id] = Status.READY


# loaded_model = load_model('model.h5')

# def analyze_query(sentence):
#

# working / service 분리 필요
def sentence_classification(sentence, project_id):
    check_time = time.time()
    sentence_morpheme = morphs(sentence)
    seq = tokenizer[project_id].texts_to_sequences([sentence_morpheme])
    train_label_vector(project_id)
    padded = pad_sequences(seq, maxlen=max_len[project_id])
    preds = model_list[project_id].predict(padded)
    print(time.time() - check_time)

    pandass = pd.Series(preds[0], Y[project_id].unique())
    pandass = pandass.sort_values(ascending=False)

    result = []
    for index, item in zip(range(3), pandass.items()):
        print(item)
        result.append((list(item)))

    return result


# working / service 분리 필요
# def sentence_similarity(sentence, threshold,project_id):
#     set = similarity(sentence, threshold,project_id)
#
#     if not set:
#         return -1
#
#     return get_key(set[0][0],project_id)


def morphs(sentence):
    me = Okt()
    clean_word = me.pos(sentence, stem=True, norm=True)
    join_word = []
    for word in clean_word:
        if not word[1] in ["Josa"]:
            join_word.append(word[0])

    document = ' '.join(join_word)

    return document
