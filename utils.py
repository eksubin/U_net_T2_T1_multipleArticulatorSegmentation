from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
from config import *


def trainGenerator(batch_size,train_path, image_folder, mask_folder, aug_dict, target_img_size, image_color="grayscale", seed=20):
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)

    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color,
        target_size=target_img_size,
        batch_size=batch_size,
        seed=seed,
    )

    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=image_color,
        target_size=target_img_size,
        batch_size=batch_size,
        seed=seed,
    )

    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        yield (img, mask)


# Test generator

def testGenerator(test_path, num_image=2, flag_multi_class=False, target_img_size=target_img_size, as_gray=True):
    files = os.listdir(test_path)
    for filename in files:
        img = io.imread(os.path.join(test_path, filename), as_gray=as_gray)
        img = img / 255
        img = trans.resize(img, target_img_size)
        img = np.reshape(img, img.shape + (1,)) if (not flag_multi_class) else img
        img = np.reshape(img, (1,) + img.shape)
        yield img

        
# saving result
def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)
        

#File path generator
def output_file_path(test_name,n_epochs,n_images,Day,Date,articulator):
    filename = test_name+"_"+n_epochs+"_"+n_images+"_"+Day+"_"+Date+".hdf5"
    model_path =os.path.join("Output/Models", articulator, filename)
    return model_path

#label visualizer

def labelVisualize(num_class,color_dict,img):
    #print('working')
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255
