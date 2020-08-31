import numpy as np
import pandas as pd
import csv
from konlpy.tag import Okt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.threading.set_inter_op_parallelism_threads(8)
tf.config.threading.set_intra_op_parallelism_threads(32)

# convert the data format(Wise iChat v2) to new data format
# new format(sentence, intent)
def csv_preprocessing(filename):
    filteredFilename = filename.split("/")[2]
    filteredCsvFile = open("./data/filtered_" + filteredFilename, 'w', encoding="utf-8", newline='')
    wr = csv.writer(filteredCsvFile)
    wr.writerow(["category", "intent", "intentType", "userSentence"])

    with open(filename, newline='', encoding='utf-8') as csvfile:
        csvReader = csv.reader(csvfile, delimiter=",")
        next(csvReader)

        for line in csvReader:
            intent = line[0].replace(" ", "_")
            category = line[1]
            intentType = line[2]
            userSentences = []
            for userSentence in line[3:]:
                if "#@#" in userSentence:
                    filteredInUserSentence = userSentence.split("#@#")[0]
                else:
                    filteredInUserSentence = userSentence

                userSentences.append(filteredInUserSentence)

            for userSentence in userSentences:
                wr.writerow([category, intent, intentType, userSentence])
            # print(intent + " | " + category + " | " + intentType + " | " +
            #       " | ".join(userSentences))
        filteredCsvFile.close()


def load_dataSet(filename):
    df = pd.read_csv(filename, encoding='utf-8', names=["Category", "Intent", "intentType", "UserSentence"], skiprows=1)
    # print(df)
    intent = df["Intent"]
    # print(intent.size())
    unique_intent = list(set(intent))
    # print(unique_intent)
    print("Intent 개수 : " + str(len(unique_intent)))
    user_sentence = df['UserSentence']
    print("사용자 질문 개수: " + str(len(user_sentence)))
    sentences = list(user_sentence)

    return intent, unique_intent, sentences


def new_load_dataSet(filename):
    df = pd.read_csv(filename, encoding='utf-8', names=["INTENT", "USERSENTENCE"], skiprows=1)
    intent = df["INTENT"].astype(str)
    unique_intent = list(set(intent))
    print("Intent 개수 : " + str(len(unique_intent)))
    user_sentence = df['USERSENTENCE']
    print("사용자 질문 개수: " + str(len(user_sentence)))
    sentences = list(user_sentence)

    return intent, unique_intent, sentences

def get_tokens(sentences, option="multi"):
    tokenizer = Okt()
    tokens = []
    if option =="multi":
        for s in sentences:
            words = tokenizer.nouns(s)
            if len(words) != 0:
                tokens.append(words)
            else:
                tokens.append([s])
    else:
        words = tokenizer.nouns(sentences)
        if len(words) != 0:
            tokens = words
        else:
            tokens.append(sentences)
    print("token 개수:" + str(len(tokens)))
    # print(tokens[:5])
    return tokens

def create_tokenizer(tokens, filters):
    tokenizer = Tokenizer(filters=filters)
    tokenizer.fit_on_texts(tokens)
    return tokenizer


def max_length(tokens):
    return len(max(tokens, key=len))

def encoding(tokenizer, tokens):
    return (tokenizer.texts_to_sequences(tokens))

def padding(encoded, max_length):
  return pad_sequences(encoded, maxlen=max_length, padding="post")

def one_hot(encode):
  o = OneHotEncoder(sparse = False)
  return(o.fit_transform(encode))

def create_model(vocab_size, max_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 128, input_length = max_length, trainable = False))
    model.add(Bidirectional(LSTM(128)))
    model.add(Dense(32, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(unique_intent), activation = 'softmax'))

    return model

def predictions(text):
    test_tokens = get_tokens(text, option="single")
    test_sequence = tokenizer.texts_to_sequences(test_tokens)
    print(test_tokens)

    if [] in test_sequence:
        test_sequence = list(filter(None, test_sequence))

    test_sequence = np.array(test_sequence).reshape(1, len(test_sequence))

    x = padding(test_sequence, max_length)
    pred = model.predict_proba(x)

    return pred

def get_final_output(pred, classes):
    predictions = pred[0]

    classes = np.array(classes)
    ids = np.argsort(-predictions)
    classes = classes[ids]
    predictions = -np.sort(-predictions)

    for i in range(pred.shape[1]):
        print("%s has confidence = %s" % (classes[i], (predictions[i])))



# parsing raw data
# csv_preprocessing("./data/intent_userSentence.csv")

# load filtered raw data
intent, unique_intent, sentences = load_dataSet("./data/filtered_intent_userSentence.csv")
# intent, unique_intent, sentences = new_load_dataSet("./data/trainQuery.tsv")

# Input Encoding
tokens = get_tokens(sentences)
tokenizer = create_tokenizer(tokens, filters='!"#$%&()*+,-/:;<=>?@[\]^`{|}~')
vocab_size = len(tokenizer.word_index) + 1
max_length = max_length(tokens)
print("Vocab Size = %d and Maximum length = %d" % (vocab_size, max_length))
encoding_doc = encoding(tokenizer, tokens)
padded_doc = padding(encoding_doc, max_length)
# print(padded_doc[:5])
print("Shape of padded docs = ", padded_doc.shape)

# Output Encoding
# output_tokenizer = create_tokenizer(unique_intent, filters='')
# print(output_tokenizer.word_index)
# encoded_output = encoding(output_tokenizer, intent)
intents = intent.tolist()
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(intents)
# print(integer_encoded[:10])
encoded_output = integer_encoded.reshape(len(integer_encoded), 1)
# encoded_output = np.array(encoded_output).reshape(len(encoded_output), 1)
print(encoded_output.shape)
output_one_hot = one_hot(encoded_output)
print(output_one_hot.shape)

# Train and Validation Setcc
train_X, val_X, train_Y, val_Y = train_test_split(padded_doc, output_one_hot, shuffle = True, test_size = 0.2)
print("Shape of train_X = %s and train_Y = %s" % (train_X.shape, train_Y.shape))
print("Shape of val_X = %s and val_Y = %s" % (val_X.shape, val_Y.shape))

# Defining Model
model = create_model(vocab_size, max_length)
model.compile(loss = "categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# model.summary()

# save model
filename = './model/new_model.h5'
# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
checkpoint = ModelCheckpoint(filename, monitor='val_acc', verbose=1, save_best_only=True, mode='min')
hits = model.fit(train_X, train_Y, epochs=10, batch_size=32, validation_data=(val_X, val_Y), callbacks=[checkpoint])

# Making Predictions
# model = load_model(filename)
# text = "동남아 허니문 추천해줘"
# pred = predictions(text)
# get_final_output(pred, unique_intent)

