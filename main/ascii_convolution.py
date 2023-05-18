import numpy as np
import os
from PIL import Image
from time import time
import tensorflow as tf

path = '/Users/harrisonward/Desktop/CS/Git/pixelator/assets'

# convert grey scale to dict to speed up look up
gscale1 = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. "
gs_dict = {index:value for (index, value) in enumerate(gscale1)}

def avg_brightness(image_array):
    try:
        return int(np.average(image_array, axis=None))
    except ValueError:
        return 0

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
def to_ascii(image, n_cols, scale, output):
    
    f = open(f'main/output/{output}', 'w')

    image_array = np.flip(convert_to_gs(np.array(image)), axis=0)

    W, H = image_array.shape

    if W > H:
        image_array = np.flip(image_array, axis=0).T

    # transpose wide images

    # tile width
    w = W / n_cols

    # tile height 
    h = w / scale

    n_rows = int(H/h)

    for i in range(n_rows):
        line = ''
        
        # pixel number times tile height
        y1 = int(i * h)
        
        # pixel one to the right
        # could probably add a % to handle the wrap around
        y2 = int((i+1)*h)

        if i == n_rows - 1:
            y2 = H

        for j in range(n_cols):
            # x tile is the pixel number times the tile width
            x1 = int(j*w)
            x2 = int((j+1)*w)

            if i == n_cols - 1:
                x2 = W
            
            # slice the image array to get the tile 
            tile = image_array[x1:x2, y1:y2]

            # get the average brightness of the tile
            avg = avg_brightness(tile)

            # look up and append the char 
            line += gs_dict[int((avg*69)/255)]
        
        # write each line to output   
        f.write(line + '\n')
 
    # cleanup
    f.close()
    
    return output

@timer
def ascii_convl(image, kernel_size, output):
    # take in the image and transform it into an array
    print('loading array')
    image_array = np.flip(convert_to_gs(np.array(image)), axis=0)

    # reshape the image so it is a valid 4D tensor for pooling
    print('reshape array')
    image_array = image_array.reshape(1, image_array.shape[0], image_array.shape[1], 1)

    # average a neighborhood of pixels to get the luminosity of each tile
    print('pooling')
    avg_pool_2d = tf.keras.layers.AveragePooling2D(pool_size=(kernel_size, kernel_size), strides=None, padding='valid')
    luminosity_array = np.asarray(avg_pool_2d(image_array))

    # map luminosity values to chars in grayscale list
    print('mapping')
    char_map = lambda i: gs_dict[int((i*69)/255)]
    mapping_function = np.vectorize(char_map)    
    ascii_array = np.squeeze(mapping_function(luminosity_array))

    # return the ascii array as text
    print('saving')
    return np.savetxt(f'output/{output}', ascii_array, fmt='%c')


def main(n_cols, scale):
    # load images
    images = []
    for file in os.listdir(path):
        fname, ftype = file.split('.')
        if fname != '' and ftype == 'jpeg':
            images.append((Image.open(f'{path}/{file}'), fname))
            print(f'File: {file}\nShape: {images[-1][0].size}')
    
    ascii_convl(images[0][0], 5, 'aaa.txt')
    # for i, image_tuple in enumerate(images):
    #     print('-'*100)
    #     print(f'Image {i + 1} of {len(images)} loading:')
    #     print(f'{image_tuple[1]}, rendering...')
    #     to_ascii(image_tuple[0], n_cols, scale, f'{image_tuple[1]}_ascii.txt')
    #     print(f'ASCII art written to output/{image_tuple[1]}_ascii.txt')




if __name__ == '__main__':
    main(600, 0.35)



