# -*- coding: cp949 -*- 

#load data
from datasets import load_dataset
import pandas as pd

dataset = load_dataset("nsmc")

train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])

print("Load Data Done!")

import re



docs = dataset['train']['document'] + dataset['test']['document']
label = dataset['train']['label'] + dataset['test']['label']

processed_docs = [re.sub("[\s]+", " ", re.sub("[^??-?Ra-zA-Z0-9]", " ", doc)) for doc in docs]

from konlpy.tag import Okt
import pickle

okt = Okt()

res = []

import os

if not os.path.isfile('senspos.pickle'):
  for doc in docs:
    tokenlist = okt.pos(doc)
    temp = []
    for w in tokenlist:
      if w[1] in ['Noun', 'Verb', 'Adjective', 'Adverb', 'Exclamation', 'Foreign', 'Alpha', 'Number', 'Unknown']:
        temp.append(w[0])
    res.append(temp)

  with open("senspos.pickle", 'wb') as f:
    pickle.dump(res, f)
  
else:
  with open("senspos.pickle", 'rb') as f:
    res = pickle.load(f)


print("POS Extraction Done!")

# padding
maxlen = max(len(x) for x in res)
padded_sens = []
for i in range(len(res)):
  sen = res[i]
  temp = sen + [" <PAD/>"] * (maxlen - len(sen))
  padded_sens.append(temp)

print("Padding Done!")

# vocab to index
import nltk
tokens = [t for d in padded_sens for t in d]
text = nltk.Text(tokens, name='NSMC')
word_count = text.vocab()
vocabulary_inv = [x[0] for x in word_count.most_common()]
vocabulary = {x:i for i, x in enumerate(vocabulary_inv)}


# padded sens to index sens
import numpy as np

x = np.array([[vocabulary[w] for w in sentence] for sentence in padded_sens])
y = np.array(label)

print("Indexing Done!")

# word2vec
from gensim.models import word2vec
import numpy as np

w2v = word2vec.Word2Vec.load("data\w2v\ko.bin")

w2v = {k: w2v[w] if w in w2v else np.random.uniform(-0.25, 0.25, w2v.vector_size) for w, k in vocabulary.items()}

print("W2V Done!")

# args setting
embedding_dim = 200
filter_sizes = (3, 4, 5)
num_filters = 100
dropout = 0.5
hidden_dims = 100

batch_size = 50
num_epochs = 10
# min_word_count = 1
# context = 10

# make dataset to use at training

X_train = x[:len(train_df)]
y_train = y[:len(train_df)]
X_test = x[len(train_df):]
y_test = y[len(train_df):]

del x, y, tokens, text, word_count, padded_sens, res, docs, label, train_df, test_df, dataset, vocabulary

# setting model and train
from tensorflow import keras

inp = keras.layers.Input(shape=(X_test.shape[1], ))

m = keras.layers.Embedding(len(vocabulary_inv), embedding_dim, input_length=X_test.shape[1], name="embedding")(inp) # non-static-cnn
m = keras.layers.Dropout(dropout)(m)

# build conv block
conv_blocks = []
for s in filter_sizes:
  conv = keras.layers.Conv1D(filters=num_filters,
                         kernel_size=s,
                         padding="valid",
                         activation="relu",
                         strides=1)(m)
  conv = keras.layers.MaxPooling1D(pool_size=2)(conv)
  conv = keras.layers.Flatten()(conv)
  conv_blocks.append(conv)
m = keras.layers.Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

m = keras.layers.Dropout(dropout)(m)
m = keras.layers.Dense(hidden_dims, activation="relu")(m)
out = keras.layers.Dense(1, activation="sigmoid")(m)

model = keras.Model(inp, out)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# add embedding layer for non-static-cnn
w = np.array([v for v in w2v.values()])
embedding_layer = model.get_layer("embedding")
embedding_layer.set_weights([w])

model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs,
          validation_data=(X_test, y_test), verbose=2)
