import numpy as np
import pickle
import time
import matplotlib.pyplot as plt

from cosine import Cosine

from keras.models import Model
from keras.layers import Dense, Input
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

batch_size = 5000
vocabulary_size = 50000
embedding_size = 128

def data_generator():
    '''
    内存有限，无法一次性将数据载入内存
    此处构造生成器，用于分批训练模型
    '''
    while True:
        with open("input_data.txt", "r", encoding="utf-8") as f:
            for line in f:
                x, y = tuple(map(lambda x:np.expand_dims(to_categorical(float(x), vocabulary_size), 0), line.strip().split(",")))
                yield ({"projection_input":x}, {"output":y})

def train_model():
    '''
    构造并训练模型
    '''
    projection_input = Input(shape=(vocabulary_size,), name="projection_input")
    projection = Dense(units=embedding_size, name="projection")(projection_input)
    output = Dense(units=vocabulary_size, activation="softmax", name="output")(projection)
    model = Model(projection_input, output)
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    history = model.fit_generator(data_generator(), steps_per_epoch=3840000//batch_size, epochs=15, verbose=2, callbacks=[
                    EarlyStopping(monitor='loss', min_delta=0.0001, patience=2, verbose=0, mode='auto', restore_best_weights=True)
                ])
    return model, history

def validate(vectors):
    '''
    验证词向量
    '''
    val_data = pickle.load(open("./data/val.p", "rb"))
    reverse_dictionary = pickle.load(open("./data/reverse_dictionary.p", "rb"))
    val_vectors = vectors[val_data]
    cosine = Cosine(topK=9)  # 为每个词语寻找9个最相似的词语，包含自身
    indices, similarities = cosine.cal_similarity(val_vectors, vectors)
    for row, index in enumerate(indices):
        word = reverse_dictionary[val_data[row]]
        neighbors = [reverse_dictionary[idx] for idx in index[1:]]  # 跳过词语本身
        print("{}: {}".format(word, neighbors))

def history_visualize(history):
    '''
    可视化训练过程
    '''
    plt.plot(history["loss"])
    plt.savefig("loss.png")

if __name__ == "__main__":
    start_time = time.time()
    model, history = train_model()  # 训练模型
    train_time = time.time()
    print("train model cost {} minutes {} seconds.".format(round(train_time-start_time)//60, round(train_time-start_time)%60))
    history_visualize(history.history)

    vectors = model.layers[1].get_weights()[0]  # 获得词向量
    with open("word_vectors.pkl", "wb") as f:
        pickle.dump(vectors, f)
    print("saved word vectors.")

    validate(vectors)
