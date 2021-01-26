from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, Flatten, Dense,ZeroPadding2D,Lambda
from tensorflow.keras import Model
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD


def mish(x):
    return Lambda(lambda x: x*K.tanh(K.softplus(x)))(x)


def buildLeNetModel(inputs, num_classes):
    X_input=Input(inputs)
    # X=ZeroPadding2D((1,1))(X_input)
    X=Conv2D(32,(5,5),strides=(1,1),padding='valid',name='conv1')(X_input)
    X=Activation(mish)(X)
    X=MaxPooling2D((2,2),strides=(2,2))(X)
    X=Conv2D(32,(5,5),strides=(1,1),padding='valid',name='conv2')(X)
    
    X=MaxPooling2D((2,2),strides=(2,2))(X)
    X=Activation(mish)(X)
    X=Flatten()(X)
    X=Dense(150,activation=mish,name='fc1')(X)
    X=Dense(num_classes,activation='softmax')(X)
    model=Model(inputs=X_input,outputs=X,name='lenet_5')

    model.summary()  

    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])

    return model

