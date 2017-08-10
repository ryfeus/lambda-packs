# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy


def handler(event, context):
    # fix random seed for reproducibility
    seed = 7
    numpy.random.seed(seed)
    # load pima indians dataset
    dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
    # split into input (X) and output (Y) variables
    X = dataset[:,0:8]
    Y = dataset[:,8]
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
    model.add(Dense(8, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model
    model.fit(X, Y, nb_epoch=150, batch_size=10,  verbose=2)
    # calculate predictions
    predictions = model.predict(X)
    # round predictions
    rounded = [round(x[0]) for x in predictions]
    print(rounded)
    return rounded