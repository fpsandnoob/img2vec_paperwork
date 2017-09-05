from keras.layers import LSTM, Embedding, Dense, Dropout, Activation, GRU
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.metrics import *
import numpy as np
import timeit
das = timeit.Timer


def train_evaluate(embedding_dims, layer_id, training_size):
    def one_hot(x_):
        index = 0
        temp = np.zeros([x_.shape[0], 12])
        for item in x_:
            temp[index][int(item) - 1] = 1
            index += 1
        return temp

    def two_layer_LSTM(dims):
        model.add(LSTM(128, input_shape=(None, dims), return_sequences=True))
        model.add(LSTM(64))

    def one_layer_GRU(dims):
        model.add(GRU(64, input_shape=(None, dims)))

    def two_layer_GRU(dims):
        model.add(GRU(128, input_shape=(None, dims), return_sequences=True))
        model.add(GRU(64))

    def one_layer_LSTM(dims):
        model.add(LSTM(64, input_shape=(None, dims)))

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    MAX_LEN = 71
    embedding = np.load('img2vec.npy')  # (8985, 1024)
    test = np.load('zh_simplified_test.npy')  # 118615
    train = np.load('zh_simplified_train.npy')  # 355843
    # print(train.shape)
    for x in test:
        x = x.split('\t')
        x_test.append(x[1].split(','))
        y_test.append(x[0])
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    for x in train:
        x = x.split('\t')
        x_train.append(x[1].split(','))
        y_train.append(x[0])
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    del test, train
    y_train = one_hot(y_train)
    y_test = one_hot(y_test)
    x_train_padded = pad_sequences(x_train, maxlen=71)
    x_test_padded = pad_sequences(x_test, maxlen=71)
    model = Sequential()
    model.add(Embedding(
        8985,
        embedding_dims,
        weights=[embedding],
        input_length=71,
        trainable=False
    ))
    if layer_id == 1:
        one_layer_GRU(embedding_dims)
    if layer_id == 2:
        two_layer_GRU(embedding_dims)
    if layer_id == 3:
        one_layer_LSTM(embedding_dims)
    if layer_id == 4:
        two_layer_LSTM(embedding_dims)

    model.add(Dense(12, input_shape=(None, 64)))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', recall_threshold(0)])

    history_result = model.fit(x_train_padded, y_train, batch_size=512, epochs=5, validation_split=1-training_size,
                               shuffle=True)
    score = model.evaluate(x_test_padded, y_test, batch_size=512)
    # with open('tet.txt', 'w') as f:
    #     pred = model.predict(x_test_padded, batch_size=2048)
    #     output = []
    #     for x in pred:
    #         output.append(np.argmax(output) + 1)
    #     f.write(output)
    # print("\ntest_score:{} test_accuracy:{}".format(score[0], score[1]), score)
    np.save('result/{}_{}.npy'.format(embedding_dims, layer_id), score)
    with open('result/{}_{}.txt'.format(embedding_dims, layer_id), 'w') as f:
        f.writelines('test_score:{} test_accuracy:{} test_recall:{}'.format(score[0], score[1], score[2]))

if __name__ == '__main__':
 train_evaluate(128, 1, 1)
