import cv2
from tensorflow.keras.layers import Input, Conv2D, Add, DepthwiseConv2D, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import tensorflow as tf

params = {'padding': 'same',
          'use_bias': True,
          'trainable': False,
          }

params6 = params | {'activation': 'relu6'}

def identity(x): return x


def conv_blocks(x, kernel_size, hidden_channels, output_channels=None, strides=1, dropout=0.):
    if output_channels is None:
        # The result of this function call and x will be Added, they better match.
        output_channels = x.shape[-1]

    x = Conv2D(hidden_channels, kernel_size=1, **params6)(x)
    x = DepthwiseConv2D(        kernel_size,   **params6, strides=strides)(x)
    if dropout:
        x = Dropout(dropout)(x)
    x = Conv2D(output_channels, kernel_size=1, **params)(x)
    return x


def common_model(input_shape=(224, 224, 3)):
    tf.keras.backend.clear_session()  # Start naming layers from one.

    input_ = Input(input_shape)

    x = Conv2D(24,      kernel_size=3, **params6, strides=2)(input_)
    x = DepthwiseConv2D(kernel_size=3, **params6)(x)
    x = Conv2D(16,      kernel_size=1, **params)(x)

    # skip will always take the shortest path.
    # long will always take the longest path.
    # x is the common path.

    skip = conv_blocks(x,    kernel_size=3, hidden_channels=64, output_channels=24, strides=2)
    long = conv_blocks(skip, kernel_size=3, hidden_channels=144)
    x = Add()([skip, long])

    skip = conv_blocks(x,    kernel_size=5, hidden_channels=144, output_channels=40, strides=2)
    long = conv_blocks(skip, kernel_size=5, hidden_channels=240)
    x = Add()([skip, long])

    skip = conv_blocks(x,    kernel_size=3, hidden_channels=240, output_channels=80, strides=2)
    long = conv_blocks(skip, kernel_size=3, hidden_channels=480)
    x = Add()([skip, long])

    skip = x
    long = conv_blocks(skip, kernel_size=3, hidden_channels=480)
    x = Add()([skip, long])

    skip = conv_blocks(x,    kernel_size=5, hidden_channels=480, output_channels=112)
    long = conv_blocks(skip, kernel_size=5, hidden_channels=672)
    x = Add()([skip, long])

    skip = x
    long = conv_blocks(skip, kernel_size=5, hidden_channels=672)
    x = Add()([skip, long])

    skip = conv_blocks(x,    kernel_size=5, hidden_channels=672, output_channels=192, strides=2, dropout=.03)
    long = conv_blocks(skip, kernel_size=5, hidden_channels=1152)
    x = Add()([skip, long])

    skip = x
    long = conv_blocks(skip, kernel_size=5, hidden_channels=1152)
    x = Add()([skip, long])

    skip = x
    long = conv_blocks(skip, kernel_size=5, hidden_channels=1152)
    x = Add()([skip, long])

    x = Conv2D(1152,    kernel_size=1, strides=1, **params6)(x)
    x = DepthwiseConv2D(kernel_size=3, strides=1, **params6)(x)
    x = GlobalAveragePooling2D()(x)

    return input_, x


def mediapipe_hand_model(weights_path=None, loss=None,  metrics=None, optimizer='adam', trainable=None):
    """The model of MediaPipe Hands that estimates the hand skeleton."""
    input_, common = common_model()

    handedness  = Dense(1,  name='conv_handedness', activation='sigmoid')(common)
    handflag    = Dense(1,  name='conv_handflag',   activation='sigmoid')(common)
    landmarks   = Dense(63, name='conv_landmarks'                       )(common)
    landmarks3D = Dense(63, name='conv_world_landmarks'                 )(common)

    model = Model(input_, [handedness, handflag, landmarks, landmarks3D])

    # Set layers as not trainable, except the specified ones.
    model = prepare_model(model, weights_path, loss, metrics, optimizer, trainable)

    return model


def closed_hand_model(weights_path=None, loss='MSE', metrics=None, optimizer='adam',
                      conv_name='conv_landmarks_closed', points=15, trainable=None):
    """Creates a model with the same structure as de MediaPipe Hands but with a different number of point landmarks and
       no handflag or handedness.

    :param weights_path: Path to a MediaPipe Hands model or to a model created with this function to load weights from.
    :param loss: Loss with which compile the model.
    :param trainable: List of names of layers to set as trainable. None is equivalent to [conv_name].
    """
    input_, common = common_model((448, 448, 3))

    landmarks = Dense(2 * points, name=conv_name)(common)

    model = Model(input_, landmarks)

    trainable = [conv_name] if trainable is None else trainable
    model = prepare_model(model, weights_path, loss, metrics, optimizer, trainable)

    return model


def opened_hand_model(weights_path=None, loss='MSE', metrics=None, optimizer='adam',
                      conv_name='conv_landmarks_opened', points=23, trainable=None):
    return closed_hand_model(weights_path, loss, metrics, optimizer, conv_name, points, trainable)


def detect_hand_model(weights_path=None, loss='MSE', metrics=None, optimizer='adam',
                        conv_name='conv_detect', trainable=None):
    """Creates a model with the same structure as de MediaPipe Hands but with a different number of point landmarks.

    :param weights_path: Path to a MediaPipe Hands model or to a model created with this function to load weights from.
    :param loss: Loss with which compile the model.
    :param trainable: List of names of layers to set as trainable.
    """
    input_, common = common_model()

    handedness = Dense(1, name='conv_handedness', activation='sigmoid')(common)
    handflag   = Dense(1, name='conv_handflag',   activation='sigmoid')(common)
    landmarks  = Dense(2 * 2, name=conv_name                     )(common)

    model = Model(input_, [handedness, handflag, landmarks])

    model = prepare_model(model, weights_path, loss, metrics, optimizer, [conv_name] if trainable is None else trainable)

    return model


def prepare_model(model, weights_path=None, loss='MSE', metrics=None, optimizer='adam', trainable=None):
    # Set layers as not trainable, except the specified ones.
    if trainable is not None:
        for index, layer in enumerate(model.layers):
            layer.trainable = layer.name in trainable or index in trainable or (index - len(model.layers)) in trainable

    if weights_path is not None:
        try:
            model.load_weights(weights_path)
        except ValueError:
            print("WARNING: Specified weights don't fully correspond with model.\n"
                  "         Loading layers by_name instead.")
            model.load_weights(weights_path, by_name=True)

    if False:
        a, b = model.layers[-1].get_weights()
        model.layers[-1].set_weights([a / 224, b / 224])
        print('Weights of last layer changed')

    loss = {'MSE': tf.keras.losses.MeanSquaredError(),
            'MeanSquaredError': tf.keras.losses.MeanSquaredError(),
            }.get(loss, loss)
    if loss is not None and optimizer is not None:
        model.compile(optimizer, loss, metrics)

    return model

