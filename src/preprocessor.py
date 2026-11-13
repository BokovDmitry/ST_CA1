import numpy as np
import cv2
import matplotlib.pyplot as plt

def apply_preprocessing(images):
    images = np.array(list(map(preprocessing, images)))
    return reshape_images_for_cnn(images)

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img
    
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img

def equalize(img):
    # This only works for grayscale images
    img = cv2.equalizeHist(img)
    return img

def reshape_images_for_cnn(data):
    return data.reshape(-1, 32, 32, 1)

def examine_random_training_image_after_preprocessing(data):
    plt.imshow(data[np.random.randint(0, len(data)-1)])
    plt.axis("off")
    plt.show()
    print(data.shape)