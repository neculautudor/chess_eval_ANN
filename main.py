import numpy
import pandas as pd
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalizationV1, BatchNormalizationV2


def main():
    number_of_rows = 15
    games_data = load_data('sample_if_main_data_file_too_big.csv', 0, number_of_rows)
    sample = games_data.sample(15)

    train_games = sample.values[:,2:]
    """the first two column from the datafile are extra so i remove it"""
    numpy.random.shuffle(train_games)

    xtrain = train_games[:,:773:]
    """bitmap of 773 bits representing a board state"""
    ytrain = train_games[:,773]
    """value labeled by stockfish to that board state(closer to 0 = black better, closer to 1 = white better"""

    model = get_model()
    model.fit(xtrain, ytrain, batch_size=256, epochs=10, verbose=1)


def get_model():
    """Get model function for special functions like relu and softmax"""
    model = Sequential()

    #first hidden
    model.add(Dense(2048, input_dim=773, activation='elu',
                kernel_initializer='random_uniform',
                bias_initializer='zeros'))
    model.add(BatchNormalizationV2(trainable=True))

    #second hidden
    model.add(Dense(2048, activation='elu',
                kernel_initializer='random_uniform',
                bias_initializer='zeros'))
    model.add(BatchNormalizationV2(trainable=True))

    #third hidden
    model.add(Dense(2048, activation='elu',
                kernel_initializer='random_uniform',
                bias_initializer='zeros'))

    #output
    model.add(Dense(1, activation='softmax',
                kernel_initializer='random_uniform',
                bias_initializer='zeros'))

    sgd = keras.optimizers.SGD(learning_rate=0.001, decay=1e-8, momentum=0.7, nesterov=True)

    model.compile(loss='mean_squared_error',
                    optimizer=sgd,
                    metrics=['mean_squared_error'])

    return model


def load_data(filename, starting_point, number_of_rows):
    data = pd.read_csv(filename, sep=',', nrows=number_of_rows, skiprows=starting_point)
    return data


if __name__ == "__main__":
    main()
    # test(300)

