import os
from secret import path

print(path)
for file in os.listdir(path):
    fname, ftype = file.split('.')
    if fname != '' and ftype != 'jpeg':
        file = os.rename(f'{path}/{file}', f"{path}/{fname}.jpeg")
