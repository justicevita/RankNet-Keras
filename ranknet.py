from keras.layers import Activation, Dense, Input, Subtract,Concatenate,Dropout
from keras.models import Model
import numpy as np

def get_rank_net(input_shape=32):
    #get two input,one is a vector regarded as more relative ,another is a vector regarded as less relative
    input_relative = Input(shape=(input_shape,), dtype='float32')
    input_unrelative = Input(shape=(input_shape,), dtype='float32')

    #get a score model which shared by two input vector
    input_layer = Input(shape=(input_shape,), dtype='float32')
    model = Dense(256, activation='relu',)(input_layer)
    model = Dense(128, activation='relu')(model)
    model = Dense(64, activation='relu')(model)
    model = Dense(32, activation='relu')(model)
    model = Dense(1)(model)
    score_model = Model(input_layer, model, name='score_model')

    #the output of score model by two vector respective
    out_relative = score_model(input_relative)
    out_unrelative = score_model(input_unrelative)

    #get the diff of two score
    subtract = Subtract()([out_relative, out_unrelative])

    #get the probability of the first input is more relative than the second input
    final_out = Activation('sigmoid')(subtract)

    #build the model
    model = Model(inputs=[input_relative, input_unrelative], outputs=final_out)
    model.compile(optimizer='rmsprop',
                       loss='binary_crossentropy',
                       metrics=['acc'])

    return model



if __name__=='__main__':
    # a simple test
    input_shape=32
    model=get_rank_net(input_shape)
    N = 100
    X_1 = 2 * np.random.uniform(size=(N, input_shape))
    X_2 = np.random.uniform(size=(N, input_shape))
    y = np.ones((X_1.shape[0], 1))

    # Train model.
    NUM_EPOCHS = 10
    BATCH_SIZE = 10
    history = model.fit([X_1, X_2], y, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, verbose=1)
