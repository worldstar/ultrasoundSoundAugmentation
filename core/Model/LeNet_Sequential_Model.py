from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, Flatten, Dense,Lambda
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam


def mish(x):
    return Lambda(lambda x: x*K.tanh(K.softplus(x)))(x)

def buildLeNetModel(img_height, img_width, img_channl, num_classes):
    inputs = (img_height, img_width, img_channl)
    model = Sequential()
    
    model.add(Conv2D(32, (5, 5), strides = (1, 1), padding = 'valid', activation = mish, input_shape = inputs))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(64, (5, 5), strides = (1, 1), padding = 'valid', activation = mish))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    # model.add(Activation(mish))
    model.add(Flatten())
    model.add(Dense(100, activation = mish))
    model.add(Dense(num_classes, activation = 'softmax'))

    model.summary()    
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])


    return model
