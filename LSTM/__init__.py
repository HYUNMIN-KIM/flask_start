import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer #convert text comment into a numeric vector
from sklearn.feature_extraction.text import TfidfTransformer #use TF IDF transformer to change text vector created by count vectorizer
from sklearn.svm import SVC# Support Vector Machine
from sklearn.pipeline import Pipeline #pipeline to implement steps in series
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC


df = pd.read_csv('tfidfDataFile.csv',delimiter=',',encoding='UTF-8')
df.head()
df.info()

# sns.countplot(df.v1)
# plt.xlabel('Label')
# plt.show()
# X = df.v2
# Y = df.v1
# le = LabelEncoder()
# Y = le.fit_transform(Y)
# Y = Y.reshape(-1,1)
X,Y =df['v2'].tolist(), df['v1'].tolist()
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15)


tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(X_train)
Train_X_Tfidf = (tfidf_vectorizer.transform(X_train)).toarray()

print("----------------")
max_words = 10
max_len = 150


sequences_matrix = sequence.pad_sequences(Train_X_Tfidf,maxlen=max_len)
print(Train_X_Tfidf)

text_clf = Pipeline([('tfidf', TfidfTransformer()),("multilabel",OneVsRestClassifier(LinearSVC(random_state=0)))])
Y_train=df[[i for i in df.columns if i not in ["v2","v1"]]]
#train model
text_clf.fit(df['v2'],Y_train)


predicted1=text_clf.predict(df['v2'])
label_cols=df.columns[2:]



# #
# def RNN():
#     inputs = Input(name='inputs',shape=[max_len])
#     layer = Embedding(max_words,50,input_length=max_len)(inputs)
#     layer = LSTM(64)(layer)
#
#     layer = Activation('relu')(layer)
#     layer = Dropout(0.5)(layer)
#     layer = Dense(1,name='out_layer')(layer)
#     layer = Activation('sigmoid')(layer)
#     model = Model(inputs=inputs,outputs=layer)
#     return model
#
# model = RNN()
# model.summary()
# model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
#
# model.fit(sequences_matrix,Y_train,batch_size=128,epochs=10,
#           validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
#
# # test_sequences = tok.texts_to_sequences(X_test)
# # test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)
#
# Test_X_Tfidf = (tfidf_vectorizer.transform(X_test)).toarray()
# test_sequences_matrix = sequence.pad_sequences(Test_X_Tfidf,maxlen=max_len)
#
# print(X_test)
#
# accr = model.evaluate(test_sequences_matrix,Y_test)

