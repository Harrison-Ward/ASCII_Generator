import os 
from secret import path

print(path)
for file in os.listdir(path):
    fname = file.split('.')[0]
    if fname is not None:
        file = os.rename(file, f"{fname}.jpg")


