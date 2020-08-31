import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import csv

from sklearn.model_selection import train_test_split
from konlpy.tag import Okt
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.svm import LinearSVR
from sklearn.svm import SVC
from sklearn.svm import OneClassSVM
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from gensim.models import Word2Vec
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import sklearn
print('sklearn: %s' % sklearn.__version__)

df = pd.read_csv('trainQuery_clean.csv',encoding='utf-8')


print(pd.DataFrame(df['id'].unique()).values)
print(pd.DataFrame(df['id'].unique()).size)
print(pd.DataFrame(df['sentence'].unique()).size)




###형태소 분석 사용해서 countvector 만들기###

for i,document in enumerate(df['sentence']):
    okt = Okt()
    clean_word = okt.pos(document,norm=True)
    join_word = []
    for word in clean_word:
       # if not word[1] in ["Josa"]:
         join_word.append(word[0])

    document = ' '.join(join_word)
    df['sentence'][i] = document

#  train_word.append(clean_word)



# tfidf = TfidfVectorizer( sublinear_tf=True)
# features = tfidf.fit_transform(df["sentence"]).toarray()
#
# countvec = CountVectorizer()
# features1 = countvec.fit_transform(df['sentence']).toarray()

# pipeline = Pipeline([
#     ('vect',countvec),
#     ('tfidf',TfidfTransformer(smooth_idf =False,sublinear_tf=True))
#
# ])
# features = pipeline.fit_transform(df["sentence"]).toarray()

# hv = HashingVectorizer(n_features=10733,norm=None)
# features = hv.fit_transform(df["sentence"]).toarray()
#
# print("hv:",features.shape)
#

# # concat_features = np.concatenate((features,features1),axis=1)
# print(features1.shape)
Y = df['id']
X = df['sentence']
# X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state= 0)
#
# models = [
#
#      LinearSVC(),
#     MultinomialNB()
#     # LogisticRegression( multi_class='auto',solver='lbfgs'),
# ]
# labels = df.id
# CV = 5
#
# cv_df = pd.DataFrame(index=range(CV * len(models)))
#
# entries = []
# for model in models:
#     model_name = model.__class__.__name__
#     accuracies = cross_val_score(model, features, labels, scoring='accuracy',cv=CV)
#     for fold_idx, accuracy in enumerate(accuracies):
#         entries.append((model_name, fold_idx, accuracy))
#
# cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
#
#
# mean_accuracy = cv_df.groupby('model_name').accuracy.mean()
# std_accuracy = cv_df.groupby('model_name').accuracy.std()
#
# acc = pd.concat([mean_accuracy, std_accuracy], axis= 1,
#           ignore_index=True)
# acc.columns = ['Mean Accuracy', 'Standard deviation']
# print(acc)
# vectorizer = HashingVectorizer(n_features=1024)
X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                    test_size=0.2,
                                                    random_state = 0)


print(X_train.shape)
print(y_train.shape)
countvec = CountVectorizer()
# pipeVectorizer = vectorizer.fit(X_train)
# pipeline_vectorizer = pipeVectorizer.transform(X_train)
pipeline = Pipeline([
    ('vect',countvec),
    ('tfidf',TfidfTransformer(smooth_idf =False,sublinear_tf=True))

])
pipeVectorizer = pipeline.fit(X_train)
pipeline_vectorizer = pipeVectorizer.transform(X_train)

model = LinearSVC(C=5.0)
model.fit(pipeline_vectorizer, y_train)
print(pd.DataFrame(X_test.unique()).size)


f = open('svm_clean.csv', 'w', encoding='utf-8', newline='')
wr = csv.writer(f)
f1 = open('LGU_input.csv', 'w', encoding='utf-8', newline='')
wr1 = csv.writer(f1)
x = np.array(list(y_train))
for i,document in enumerate(X_train):
    s = document
    flag = True
    preds= model.predict(pipeVectorizer.transform([s]))
    if preds != x.flatten()[i] :
        flag = False

    wr1.writerow([x.flatten()[i],document])

x = np.array(list(y_test))
for i,document in enumerate(X_test):
    s = document
    flag = True
    preds= model.predict(pipeVectorizer.transform([s]))
    if preds != x.flatten()[i] :
        flag = False

    wr.writerow([x.flatten()[i],preds,flag,document])

f.close()
pred = model.predict(pipeVectorizer.transform(X_test))
print('예측 정확도: {0:.4f}'.format(accuracy_score(y_test,pred)))
# new_sentence="전기요금 과소청구에 의한 갑작스런 부담에 대한 구제 방법이 있나요?"
# print(model.predict(pipeVectorizer.transform([new_sentence])))