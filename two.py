import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm

Dir = 'PetImages'
CATEGORIES = ["Dog", "Cat"]
c1 = 2
for c in CATEGORIES:
    path = os.path.join(Dir, c)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array, cmap="gray")
        plt.show()
        if c1>0:
            c1-=1
        else:
            break
    break

img_size = 50
new_array = cv2.resize(img_array, (img_size, img_size))
plt.imshow(new_array, cmap ='gray')
plt.show()

training_data = []

cc = 0
def create_training_data():
    for category in CATEGORIES:  # do dogs and cats

        path = os.path.join(DATADIR,category)  # create path to dogs and cats
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat

        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            print("count:" + str(cc))
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (img_size,img_size))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))

create_training_data()

print(len(training_data))