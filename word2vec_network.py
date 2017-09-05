import numpy as np
from gensim.models import Word2Vec
from keras.layers import LSTM, Embedding, Dense, Dropout, Activation
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
def one_hot(x_):
    index = 0
    temp = np.zeros([x_.shape[0], 12])
    for item in x_:
        temp[index][int(item) - 1] = 1
        index += 1
    return temp
bin_fname = 'word2vec/word2vec_wx'
model = Word2Vec.load(bin_fname)
EMBEDDING_DIMS = 256
d = {}
for split in ['train', 'test', 'val']:
    x = []
    y = []
    src_name = 'data/zh_simplified_{}.txt'.format(split)
    with open(src_name, 'r', encoding='utf-8') as f:
        sequence = f.readline()
        while sequence is not '':
            New_seq = []
            sequence = sequence.split('\t')
            sequence[1] = sequence[1].replace('\n', '')
            for char in sequence[1]:
                # New_seq.append(model[char])
                if char not in d.keys():
                    d[char] = len(d.keys())
            x.append(New_seq)
            y.append(sequence[0])
            sequence = f.readline()
for split in ['train', 'test', 'val']:
    x = []
    y = []
    src_name = 'data/zh_simplified_{}.txt'.format(split)
    with open(src_name, 'r', encoding='utf-8') as f:
        sequence = f.readline()
        while sequence is not '':
            New_seq = []
            sequence = sequence.split('\t')
            sequence[1] = sequence[1].replace('\n', '')
            for char in sequence[1]:
                New_seq.append(d[char])
            x.append(New_seq)
            y.append(sequence[0])
            sequence = f.readline()
    np.save('zh_simplified_word2vec_{}_{}.npy'.format(split, 'data'), x)
    np.save('zh_simplified_word2vec_{}_{}.npy'.format(split, 'label'), y)
    print(np.array(x).shape)
    print(np.array(y).shape)
embedding_matrix = np.zeros((len(d) + 1, EMBEDDING_DIMS))
model = model.wv
for word, i in d.items():
    if word in model:
        embedding_matrix[i] = model[word]
del d, model
x_train = np.load('zh_simplified_word2vec_train_data.npy')
y_train = np.load('zh_simplified_word2vec_train_label.npy')
x_test = np.load('zh_simplified_word2vec_test_data.npy')
y_test = np.load('zh_simplified_word2vec_test_label.npy')
x_train_padded = pad_sequences(x_train, maxlen=71)
x_test_padded = pad_sequences(x_test, maxlen=71)
y_train = one_hot(y_train)
y_test = one_hot(y_test)
model = Sequential()
model.add(Embedding(
    8986,
    256,
    weights=[embedding_matrix],
    input_length=71,
    trainable=False
))
model.add(LSTM(128, input_shape=(None, 256), return_sequences=True))
model.add(LSTM(64))
model.add(Dense(12, input_shape=(None, 64)))
model.add(Dropout(0.5))
model.add(Activation('relu'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train_padded, y_train, batch_size=1024, nb_epoch=5, validation_split=0.33)
score = model.evaluate(x_test_padded, y_test, batch_size=1024)
print("test: ", score)

