import numpy as np # linear algebra
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
from konlpy.tag import Okt
from pyjarowinkler import distance
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

data = pd.read_csv('trainQuery_clean.csv',encoding='utf-8')

Y = data['CATEGORY']
X = data['TITLE']
Z = copy.deepcopy(X)
K = copy.deepcopy(Y)

dense_num = data['CATEGORY'].unique().size

result = {}

for i,document in enumerate(X):
    me = Okt()
    clean_word = me.pos(document,stem=True,norm=True)
    join_word  = []
    for word in clean_word:
       if not word[1] in ["Josa"]:
            join_word.append(word[0])

    document = ' '.join(join_word)
    data['TITLE'][i] = document

dicts_df = (dict(data.groupby('CATEGORY')['TITLE'].apply(list)))
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
print("max len : " ,max_len)
X = pad_sequences(sequences,max_len)
X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                    test_size=0.2,
                                                    random_state = 0)

X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(data['TITLE'], data['CATEGORY'].values,
                                                    test_size=0.2,
                                                    random_state = 0)

num_classes = dense_num
print('카테고리 : {}'.format(num_classes))

print(X_train[0]) # 첫번째 문장데이터
print(y_train[0]) # 첫번째 문장데이터 레이블

y_train_one = to_categorical(y_train) # 훈련용 원-핫 인코딩
y_test_one = to_categorical(y_test) # 테스트용 원-핫 인코딩

def similarity(query, dicts_df_l, threshold):
    query_dict = {}
    start = time.time()
    for v in dicts_df_l:

        score = distance.get_jaro_distance(v, query)

        if score >= threshold:
            # map에 집어넣기
            d1 = {v :  score}
            query_dict.update(d1)
            # map에서 가장 높은 놈 topk 만큼 뽑기


    result = sorted(query_dict.items(),reverse=True,key=operator.itemgetter(1))

    return result

start = time.time()
result = similarity("채권 혼합 공탁 경우 공탁 금 추다 급 확인 어떻다 진행 되다 ?",data["TITLE"], 0.8)


def get_key(val):
    for key, value in dicts_df.items():

        if val in value:
            return key

    return -1

print(get_key(result[0][0]))
print(time.time() - start)

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
    #
    model.add(Dense(dense_num, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    return model


cnn_model().summary()
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
#history_LSTM = cnn_model().fit(X_train, y_train_one, batch_size=128, epochs=30, callbacks=[es, mc], validation_data=(X_test, y_test_one))


#plt.plot(history_LSTM.history["loss"])
#plt.title("loss")
#plt.show()

labels = []
for i,index in enumerate(y_train.unique()):
    labels.append(index)


new_complaint = ['공제 이용중 사업자등록을 하게되면 이용중 공제 중도해지 해야하나요??']
seq = tokenizer.texts_to_sequences(new_complaint)
padded = pad_sequences(seq)



loaded_model = load_model('best_model.h5')
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test_one)[1]))

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
x = np.array(list(y_test_val))
cnt =0

for i,document in enumerate(X_test_val):

    og = document
    s = [document]
    # print(s)
    flag = True
    seq = tokenizer.texts_to_sequences(s)
    padded = pad_sequences(seq, maxlen=max_len)
    preds = loaded_model.predict(padded)
#    print(np.argmax(preds))
    type = "cnn"
    label = (np.argmax(preds))

    # print(label)
    # print(x.flatten()[i])
    # print(np.argmax(preds))
    similarity_label =  similarity(og,X_train_val,0.85)

    if not similarity_label :
        preds_label = -1
    else:
        preds_label =  get_key(similarity_label[0][0])


    if x.flatten()[i] == preds_label:
        flag = True
        type = "similarity"
        print(type)


    elif label != x.flatten()[i]:
        flag = False
        cnt+=1


    wr.writerow([x.flatten()[i], label, flag, document,type])



f.close()
print(cnt)
print(len(X_test_val))
print((len(X_test_val) - cnt) /len(X_test_val)  )
print("exit")