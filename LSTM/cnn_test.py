import numpy as np # linear algebra
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
from konlpy.tag import Okt
from Levenshtein import _levenshtein
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding,Conv1D, MaxPooling1D, GlobalMaxPooling1D, Flatten
from tensorflow.keras.preprocessing.sequence import pad_sequences
import csv
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
import copy
import operator
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



data = pd.read_csv('data/trainQuery.csv',encoding='utf-8')
data2 = pd.read_csv("data/testQuery.csv", encoding='utf-8')

testX = data2["TITLE"]
testY = data2["CATEGORY"]
Y = data['CATEGORY']
X = data['TITLE']
testZ  = copy.deepcopy(testX)
Z = copy.deepcopy(X)
K = copy.deepcopy(Y)

dicts_df = (dict(data.groupby('CATEGORY')['TITLE'].apply(list)))
dicts = copy.deepcopy(dicts_df)
dense_num = data['CATEGORY'].unique().size

result = {}


# for i,document in enumerate(X):
#     me = Okt()
#     clean_word = me.pos(document,stem=True,norm=True)
#     join_word  = []
#     for word in clean_word:
#        if not word[1] in ["Josa"]:
#             join_word.append(word[0])
#
#     document = ' '.join(join_word)
#     data['TITLE'][i] = document

# for i, document in enumerate(testX):
#       me = Okt()
#       clean_word = me.pos(document, stem=True, norm=True)
#       join_word = []
#       for word in clean_word:
#           if not word[1] in ["Josa"]:
#               join_word.append(word[0])
#
#       document = ' '.join(join_word)
#       data2['TITLE'][i] = document

print(len(dicts_df.keys()))

n_most_common_words = 5000

tokenizer = Tokenizer(num_words=n_most_common_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(data['TITLE'].values)
word_index = tokenizer.word_index

print(word_index)
print(tokenizer.word_counts)
sequences = tokenizer.texts_to_sequences(data['TITLE'].values)

print('Found %s unique tokens.' % len(word_index))
max_len =max(len(l) for l in sequences)
print("maxlent : " ,max_len)
X = pad_sequences(sequences,max_len)




num_classes = dense_num
print('카테고리 : {}'.format(num_classes))

print(X[0]) # 첫번째 문장데이터
print(Y[0]) # 첫번째 문장데이터 레이블

y_train_one = to_categorical(Y) # 훈련용 원-핫 인코딩
print(y_train_one)
def jaro(sentence, query):
    score = _levenshtein.ratio(sentence, query)
    return score

def similarity(query, dicts_df_l, threshold):
    query_dict = {}
    for v in dicts_df_l:

        score = jaro(v,query)

        if score >= threshold:
             # map에 집어넣기
            query_dict.update({v  :  score})

            # map에서 가장 높은 놈 topk 만큼 뽑기
    result = sorted(query_dict.items(),reverse=True,key=operator.itemgetter(1))[:5]



    # print("simiarity time", time.time() - start)
    return result

# def sim_sentence(query, list, threshold):
#     start = time.time()
#     result = [v for v in list if distance.get_jaro_distance(v, query) >= threshold]
#     print("simiarity time", time.time() - start)
#
#     return result

start = time.time()

print(time.time() - start)

def get_key(val):
    for key, value in dicts.items():

        if val in value:
            return key

    return -1

set = similarity("인천공항으로 가려면",Z,0.85)
print(set)
print(set[0][0])
print(get_key(set[0][0]))

def lstm_model():

    model = Sequential()

    model.add(Embedding(len(word_index)+1,100, input_length = max_len))
    model.add(LSTM(max_len))
    model.add(Dense(dense_num, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return model

def cnn_model():
    model = Sequential()

    model.add(Embedding(len(word_index)+1, 128, input_length=max_len))
    model.add(Conv1D(filters=512, kernel_size=2, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Flatten())

    model.add(Dense(dense_num, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    return model


cnn_model().summary()
es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='acc', mode='max', verbose=1, save_best_only=True)
check_time = time.time()
# cnn_model().save('cnn_model.h5')
# history_LSTM = cnn_model().fit(X, y_train_one, batch_size=128, epochs=15, callbacks=[es, mc])

print("학습시간",time.time() - check_time)

#plt.plot(history_LSTM.history["loss"])
#plt.title("loss")
#plt.show()

labels = []
for i,index in enumerate(Y.unique()):
    labels.append(index)


new_complaint = '주차료 너무비싸 할인 가능해?'
seq = tokenizer.texts_to_sequences(new_complaint)
padded = pad_sequences(seq)


loaded_model = load_model('best_model.h5')
# print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(testX, y_test_one)[1]))
preds = loaded_model.predict(padded)
print("np : " ,np.argmax(preds))
f = open('cnnfilter256_지식정제 전체 .csv', 'w', encoding='utf-8', newline='')
wr = csv.writer(f)
# f1 = open('LGU_input.csv', 'w', encoding='utf-8', newline='')
# wr1 = csv.writer(f1)
# x = np.array(list(y_train))
# for i,document in enumerate(X_train_val):
#     s = [document]
#     # print(s)
#     flag = True
#     seq = tokenizer.texts_to_sequences(s)
#     padded = pad_sequences(seq, maxlen=20)
#     preds = model.predict(padded)
#     label = labels[np.argmax(preds)]
#     if label != x.flatten()[i]:
#         flag = False
#
#     wr1.writerow([x.flatten()[i],label,flag])
#
x = np.array(list(testY))
cnt =0

print("평가 시작")
check_time = time.time()
for i,document in enumerate(testX):

    og = document
    s = [document]
    # print(s)
    flag = False
    seq = tokenizer.texts_to_sequences(s)
    padded = pad_sequences(seq, maxlen=max_len)
    preds = loaded_model.predict(padded)
    print('np :', np.argmax(preds))
    type = "none"
    label = (np.argmax(preds))

    # print(label)
    # print(x.flatten()[i])
    # print(np.argmax(preds))
    similarity_label =  similarity(testZ[i],Z,0.0)

    if not similarity_label :
        preds_label = -1
    else:
        preds_label =  get_key(similarity_label[0][0])
        type = "similarity"

    sim_score = {}

    if x.flatten()[i] == preds_label:
        flag = True
        type = "similarity"
        sim_score = similarity_label
        print(type)


    elif label == x.flatten()[i]:
        flag = True
        type = "cnn"

        print(type)

    else:
        cnt += 1

    if type == "similarity":
        label = preds_label


    wr.writerow([testZ[i],label,x.flatten()[i],type,sim_score,flag])


f.close()
print("학습시간",time.time() - check_time)
print(cnt)
print(len(testX))
print((len(testX) - cnt) /len(testX)  )
print("exit")