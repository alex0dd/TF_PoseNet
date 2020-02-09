import numpy as np

def sigmoid(x):                                        
    return 1 / (1 + np.exp(-x))

def argmax2d(t):
    if len(t.shape) > 3:
        t = np.squeeze(t)

    if not len(t.shape) == 3:
        print("Input must be a 3D matrix, or be able to be squeezed into one.")
        return

    height, width, depth = t.shape

    reshaped_t = np.reshape(t, [height * width, depth])
    argmax_coords = np.argmax(reshaped_t, axis=0)
    y_coords = argmax_coords // width
    x_coords = argmax_coords % width

    return np.concatenate([np.expand_dims(y_coords, 1), np.expand_dims(x_coords, 1)], axis=1)