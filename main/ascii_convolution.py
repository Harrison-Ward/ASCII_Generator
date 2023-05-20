import numpy as np
import os
from PIL import Image
from time import time
from tensorflow.keras.layers import AveragePooling2D

path = '/Users/harrisonward/Desktop/CS/Git/pixelator/assets'


gscale = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. "

# convert grey scale to dict to speed up look up, invert it for negative images
gs_dict = {index: value for (index, value) in enumerate(gscale)}
gs_dict_inv = {index: value for (index, value) in enumerate(reversed(gscale))}
    
def avg_brightness(image_array):
    return int(np.average(image_array, axis=None))


def convert_to_gs(image_array):
    return np.mean(np.asarray(image_array), axis=2)


def timer(func):
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Image generated in {(t2-t1):.4f}s')
        return result
    return wrap_func


@timer
def ascii_conv(image, kernel_size, output, invert: bool, negative: bool):
    # take in the image and transform it into an array
    image_array = convert_to_gs(np.array(image))

    if invert:
        if image_array.shape[1] > image_array.shape[0]:
            image_array = image_array.T

    # reshape the image so it is a valid 4D tensor for pooling
    image_array = image_array.reshape(
        1, image_array.shape[0], image_array.shape[1], 1)

    # average a neighborhood of pixels to get the luminosity of each tile
    # avg_pool_2d = tf.keras.layers.AveragePooling2D(pool_size=(
    #     kernel_size, kernel_size), strides=None, padding='valid')
    avg_pool_2d = AveragePooling2D(pool_size=(
        kernel_size, kernel_size), strides=None, padding='valid')
    luminosity_array = np.asarray(avg_pool_2d(image_array))

    # map luminosity values to chars in grayscale list
    if negative:
        def char_map(i): return gs_dict_inv[int((i*69)/255)]
    else:
        def char_map(i): return gs_dict[int((i*69)/255)]

    mapping_function = np.vectorize(char_map)
    ascii_array = np.squeeze(mapping_function(luminosity_array))

    # return the ascii array as text
    return np.savetxt(f'output/{output}', ascii_array, fmt='%c')


def main(kernel_size, invert=None, negative=None):
    # load and lazily store images
    images = []
    for file in os.listdir(path):
        fname, ftype = file.split('.')
        if fname != '' and ftype == 'jpeg':
            images.append((Image.open(f'{path}/{file}'), fname))

    for i, image_tuple in enumerate(images):
        print('-'*100)
        print(f'Image {i + 1} of {len(images)} loading:')
        print(f'{image_tuple[1]}, of size {image_tuple[0].size} rendering...')
        ascii_conv(image_tuple[0], kernel_size,
                    f'{image_tuple[1]}_ascii.txt', invert, negative)
        print(f'ASCII art written to output/{image_tuple[1]}_ascii.txt')


if __name__ == '__main__':
    main(5, invert=False, negative=True)
