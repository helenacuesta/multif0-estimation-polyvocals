'''This script contains the network architectures used in the project.
'''


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Concatenate, Conv2D, BatchNormalization
from tensorflow.keras import backend as K


def build_model1():
    '''Early/Deep
    '''

    input_shape_1 = (None, None, 5) # HCQT input shape
    input_shape_2 = (None, None, 5)  # phase differentials input shape

    inputs1 = Input(shape=input_shape_1)
    inputs2 = Input(shape=input_shape_2)

    b1a = BatchNormalization()(inputs1)
    b1b = BatchNormalization()(inputs2)

    # conv1 with hcqt
    y1a = Conv2D(16, (5, 5), padding='same', activation='relu', name='conv1a')(b1a)
    y1abn = BatchNormalization()(y1a)

    # conv1 with phase differentials
    y1b = Conv2D(16, (5, 5), padding='same', activation='relu', name='conv1b')(b1b)
    y1bbn = BatchNormalization()(y1b)

    # concatenate features
    y1c = Concatenate()([y1abn, y1bbn])

    # conv2 layer
    y2 = Conv2D(32, (5, 5), padding='same', activation='relu', name='conv2')(y1c)
    y2a = BatchNormalization()(y2)

    # conv3 layer
    y3 = Conv2D(32, (5, 5), padding='same', activation='relu', name='conv3')(y2a)
    y3a = BatchNormalization()(y3)

    # conv4 layer
    y4 = Conv2D(32, (5, 5), padding='same', activation='relu', name='conv4')(y3a)
    y4a = BatchNormalization()(y4)

    # conv5 layer, harm1
    y5 = Conv2D(32, (70, 3), padding='same', activation='relu', name='harm1')(y4a)
    y5a = BatchNormalization()(y5)

    # conv6 layer, harm2
    y6 = Conv2D(32, (70, 3), padding='same', activation='relu', name='harm2')(y5a)
    y6a = BatchNormalization()(y6)

    # conv7 layer
    y7 = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv7')(y6a)
    y7a = BatchNormalization()(y7)

    # conv8 layer
    y8 = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv8')(y7a)
    y8a = BatchNormalization()(y8)

    y9 = Conv2D(8, (360, 1), padding='same', activation='relu', name='distribution')(y8a)
    y9a = BatchNormalization()(y9)

    y10 = Conv2D(1, (1, 1), padding='same', activation='sigmoid', name='squishy')(y9a)
    predictions = Lambda(lambda x: K.squeeze(x, axis=3))(y10)

    model = Model(inputs=[inputs1, inputs2], outputs=predictions)

    return model

def build_model2():
    '''Early/Shallow
    '''

    input_shape_1 = (None, None, 5) # HCQT input shape
    input_shape_2 = (None, None, 5)  # phase differentials input shape

    inputs1 = Input(shape=input_shape_1)
    inputs2 = Input(shape=input_shape_2)

    b1a = BatchNormalization()(inputs1)
    b1b = BatchNormalization()(inputs2)

    # conv1 with hcqt
    y1a = Conv2D(16, (5, 5), padding='same', activation='relu', name='conv1a')(b1a)
    y1abn = BatchNormalization()(y1a)

    # conv1 with phase differentials
    y1b = Conv2D(16, (5, 5), padding='same', activation='relu', name='conv1b')(b1b)
    y1bbn = BatchNormalization()(y1b)

    # concatenate features
    y1c = Concatenate()([y1abn, y1bbn])

    # conv2 layer
    y2 = Conv2D(32, (5, 5), padding='same', activation='relu', name='conv2')(y1c)
    y2a = BatchNormalization()(y2)

    # conv3 layer
    y3 = Conv2D(32, (5, 5), padding='same', activation='relu', name='conv3')(y2a)
    y3a = BatchNormalization()(y3)

    # conv4 layer
    y4 = Conv2D(32, (5, 5), padding='same', activation='relu', name='conv4')(y3a)
    y4a = BatchNormalization()(y4)

    # conv5 layer, harm1
    y5 = Conv2D(32, (70, 3), padding='same', activation='relu', name='harm1')(y4a)
    y5a = BatchNormalization()(y5)

    # conv6 layer, harm2
    y6 = Conv2D(32, (70, 3), padding='same', activation='relu', name='harm2')(y5a)
    y6a = BatchNormalization()(y6)


    y7 = Conv2D(8, (360, 1), padding='same', activation='relu', name='distribution')(y6a)
    y7a = BatchNormalization()(y7)

    y8 = Conv2D(1, (1, 1), padding='same', activation='sigmoid', name='squishy')(y7a)
    predictions = Lambda(lambda x: K.squeeze(x, axis=3))(y8)

    model = Model(inputs=[inputs1, inputs2], outputs=predictions)

    return model

def build_model1_pf():

    input_shape = (None, None, 5) # HCQT input shape

    inputs = Input(shape=input_shape)

    b1a = BatchNormalization()(inputs)

    # conv1 with hcqt
    y1a = Conv2D(16, (5, 5), padding='same', activation='relu', name='conv1a')(b1a)
    y1c = BatchNormalization()(y1a)


    # conv2 layer
    y2 = Conv2D(32, (5, 5), padding='same', activation='relu', name='conv2')(y1c)
    y2a = BatchNormalization()(y2)

    # conv3 layer
    y3 = Conv2D(32, (5, 5), padding='same', activation='relu', name='conv3')(y2a)
    y3a = BatchNormalization()(y3)

    # conv4 layer
    y4 = Conv2D(32, (5, 5), padding='same', activation='relu', name='conv4')(y3a)
    y4a = BatchNormalization()(y4)

    # conv5 layer, harm1
    y5 = Conv2D(32, (70, 3), padding='same', activation='relu', name='harm1')(y4a)
    y5a = BatchNormalization()(y5)

    # conv6 layer, harm2
    y6 = Conv2D(32, (70, 3), padding='same', activation='relu', name='harm2')(y5a)
    y6a = BatchNormalization()(y6)

    # conv7 layer
    y7 = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv7')(y6a)
    y7a = BatchNormalization()(y7)

    # conv8 layer
    y8 = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv8')(y7a)
    y8a = BatchNormalization()(y8)

    y9 = Conv2D(8, (360, 1), padding='same', activation='relu', name='distribution')(y8a)
    y9a = BatchNormalization()(y9)

    y10 = Conv2D(1, (1, 1), padding='same', activation='sigmoid', name='squishy')(y9a)
    predictions = Lambda(lambda x: K.squeeze(x, axis=3))(y10)

    model = Model(inputs=inputs, outputs=predictions)

    return model

def build_model2_pf():

    input_shape = (None, None, 5)  # HCQT input shape

    inputs = Input(shape=input_shape)

    b1a = BatchNormalization()(inputs)

    # conv1 with hcqt
    y1a = Conv2D(16, (5, 5), padding='same', activation='relu', name='conv1a')(b1a)
    y1c = BatchNormalization()(y1a)

    # conv2 layer
    y2 = Conv2D(32, (5, 5), padding='same', activation='relu', name='conv2')(y1c)
    y2a = BatchNormalization()(y2)

    # conv3 layer
    y3 = Conv2D(32, (5, 5), padding='same', activation='relu', name='conv3')(y2a)
    y3a = BatchNormalization()(y3)

    # conv4 layer
    y4 = Conv2D(32, (5, 5), padding='same', activation='relu', name='conv4')(y3a)
    y4a = BatchNormalization()(y4)

    # conv5 layer, harm1
    y5 = Conv2D(32, (70, 3), padding='same', activation='relu', name='harm1')(y4a)
    y5a = BatchNormalization()(y5)

    # conv6 layer, harm2
    y6 = Conv2D(32, (70, 3), padding='same', activation='relu', name='harm2')(y5a)
    y6a = BatchNormalization()(y6)

    y7 = Conv2D(8, (360, 1), padding='same', activation='relu', name='distribution')(y6a)
    y7a = BatchNormalization()(y7)

    y8 = Conv2D(1, (1, 1), padding='same', activation='sigmoid', name='squishy')(y7a)
    predictions = Lambda(lambda x: K.squeeze(x, axis=3))(y8)

    model = Model(inputs=inputs, outputs=predictions)

    return model

def base_model(input, let):

    b1 = BatchNormalization()(input)

    # conv1
    y1 = Conv2D(16, (5, 5), padding='same', activation='relu', name='conv1{}'.format(let))(b1)
    y1a = BatchNormalization()(y1)

    # conv2
    y2 = Conv2D(32, (5, 5), padding='same', activation='relu', name='conv2{}'.format(let))(y1a)
    y2a = BatchNormalization()(y2)

    # conv3
    y3 = Conv2D(32, (5, 5), padding='same', activation='relu', name='conv3{}'.format(let))(y2a)
    y3a = BatchNormalization()(y3)

    # conv4 layer
    y4 = Conv2D(32, (5, 5), padding='same', activation='relu', name='conv4{}'.format(let))(y3a)
    y4a = BatchNormalization()(y4)

    # conv5 layer, harm1
    y5 = Conv2D(32, (70, 3), padding='same', activation='relu', name='harm1{}'.format(let))(y4a)
    y5a = BatchNormalization()(y5)

    # conv6 layer, harm2
    y6 = Conv2D(32, (70, 3), padding='same', activation='relu', name='harm2{}'.format(let))(y5a)
    y6a = BatchNormalization()(y6)

    return y6a, input


def build_model3():
    '''Late/Deep
    '''

    input_shape_1 = (None, None, 5) # HCQT input shape
    input_shape_2 = (None, None, 5)  # phase differentials input shape

    inputs1 = Input(shape=input_shape_1)
    inputs2 = Input(shape=input_shape_2)

    y6a, _ = base_model(inputs1, 'a')
    y6b, _ = base_model(inputs2, 'b')

    # concatenate features
    y6c = Concatenate()([y6a, y6b])

    # conv7 layer
    y7 = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv7')(y6c)
    y7a = BatchNormalization()(y7)

    # conv8 layer
    y8 = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv8')(y7a)
    y8a = BatchNormalization()(y8)

    y9 = Conv2D(8, (360, 1), padding='same', activation='relu', name='distribution')(y8a)
    y9a = BatchNormalization()(y9)

    y10 = Conv2D(1, (1, 1), padding='same', activation='sigmoid', name='squishy')(y9a)
    predictions = Lambda(lambda x: K.squeeze(x, axis=3))(y10)

    model = Model(inputs=[inputs1, inputs2], outputs=predictions)

    return model

def build_model3_mag():
    '''Late/Deep no phase
    '''

    input_shape_1 = (None, None, 5) # HCQT input shape

    inputs1 = Input(shape=input_shape_1)

    y6a, _ = base_model(inputs1, 'a')

    # conv7 layer
    y7 = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv7')(y6a)
    y7a = BatchNormalization()(y7)

    # conv8 layer
    y8 = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv8')(y7a)
    y8a = BatchNormalization()(y8)

    y9 = Conv2D(8, (360, 1), padding='same', activation='relu', name='distribution')(y8a)
    y9a = BatchNormalization()(y9)

    y10 = Conv2D(1, (1, 1), padding='same', activation='sigmoid', name='squishy')(y9a)
    predictions = Lambda(lambda x: K.squeeze(x, axis=3))(y10)

    model = Model(inputs=inputs1, outputs=predictions)

    return model