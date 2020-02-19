from keras import backend as K


def sobel_x_init(shape, dtype=None):
    return K.constant([-1, 0, 1, -2, 0, 2, -1, 0, 1], dtype, shape)


def sobel_y_init(shape, dtype=None):
    return K.constant([-1, -2, -1, 0, 0, 0, 1, 2, 1], dtype, shape)