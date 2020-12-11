import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
import cv2,os
from core.Model.LeNet import buildLeNetModel

class CustomDataGenerator(ImageDataGenerator):
  def __init__(self,
               clahenum=None,
               **kwargs):
    '''
    Custom image data generator.
    Behaves like ImageDataGenerator, but allows color augmentation.
    '''
    super().__init__(
        preprocessing_function=self.augment_color,
        **kwargs)

    self.clahe_num=clahenum

  def augment_color(self, image):
    '''Takes an input image and returns a modified version of it'''
    temp=random.randint(0, 40)
    img = image*255 # convert to numpy array in range 0-255
    img = img.astype(np.uint8) # convert to int


    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)  
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(self.clahe_num,self.clahe_num))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # final_t = np.expand_dims(img, axis=-1)
    f=('./test/{}.png'.format(temp))
    cv2.imwrite(f,final)

    return final

if __name__ == "__main__":
    readDataPathTrain = "D:/Desktop/ga/imageAugmentation/data/Train/"
    readDataPathVal = "D:/Desktop/ga/imageAugmentation/data/val/"
    img_height, img_width, img_channl = (256,256,3) #224, 224, 3
    batch_size=32
    epochs=10
    num_classes = 7


    datagen=CustomDataGenerator(clahenum=40,dtype=int)

    train_generator = datagen.flow_from_directory(
        readDataPathTrain,
        target_size=(256, 256),
        batch_size=32,
        class_mode='binary')

    val_generator = datagen.flow_from_directory(
        readDataPathVal,
        target_size=(256, 256),
        batch_size=32,
        class_mode='binary')

    model = buildLeNetModel(img_height, img_width, img_channl, num_classes)

    model.fit(
        train_generator,
        steps_per_epoch=10,
        epochs=50,
        validation_data=val_generator,
        validation_steps=10)
