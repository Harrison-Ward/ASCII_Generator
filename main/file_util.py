import os
from secret import path

print(path)
for file in os.listdir(path):
    fname, ftype = file.split('.')
    if fname is not None and ftype != 'jpg':
        file = os.rename(file, f"{fname}.jpg")
