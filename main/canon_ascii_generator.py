import numpy as np
import os
from PIL import Image

gscale1 = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. "
path = '/Users/harrisonward/Desktop/CS/Git/pixelator/assets'

images = []
for file in os.listdir(path):
    fname, ftype = file.split('.')
    if fname != '' and ftype == 'jpeg':
        images.append(Image.open(f'{path}/{file}'))

def img_to_ascii(input_image, n_cols, scale):
    output_image = []
    
    W, H = input_image.size[0], input_image.size[1]
    
    # tile width
    w = W / n_cols

    # tile height 
    h = w / scale

    n_rows = int(H/h)

    print(f"cols: {n_cols}, rows: {n_rows}")
    print(f"tile dims: {w} x {h}")
 
    # check if image size is too small
    if n_cols > W or n_rows > H:
        print("Image too small for specified cols!")

    for i in range(n_rows):
        
        # pixel number times tile height
        y1 = int(i * h)
        
        # pixel one to the right
        # could probably add a % to handle the wrap around
        y2 = int((i+1)*h)

        if i == n_rows - 1:
            y2 = H

        # add an empty string 
        output_image.append("")

        for j in range(n_cols):
            # x tile is the pixel number times the tile width
            x1 = int(j*w)
            x2 = int((j+1)*w)

            if i == n_cols - 1:
                x2 = W
            
            # crop the image, get the tile 
            # could also take a slice of the matrix at these indicies
            img = input_image.crop((x1, y1, x2, y2))

            # get the average brightness
            # avg = int(avg_brightness(img))
            avg = int(getAverageL(img))

            # look up ascii char
            gsval = gscale1[int((avg*69)/255)]

            # append the char 
            output_image[i] += gsval
    
    return output_image

def getAverageL(image):
 
    """
    Given PIL Image, return average value of grayscale value
    """
    # get image as numpy array
    im = np.array(image)
 
    # get shape
    w,h = im.shape
 
    # get average
    return np.average(im.reshape(w*h))


def main(image, n_cols, scale):
    output_image = img_to_ascii(image, n_cols, scale)
    
    print('generating ASCII art...')
    
    # convert image to ascii txt
    outFile = 'out.txt'
 
    # open file
    f = open(outFile, 'w')
 
    # write to file
    for row in output_image:
        f.write(row + '\n')
 
    # cleanup
    f.close()
    print("ASCII art written to %s" % outFile)

if __name__ == '__main__':
    main(images[2].convert('L'), n_cols=200, scale=0.43)