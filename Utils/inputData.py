from keras.preprocessing.image import ImageDataGenerator
from Utils import config
import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
from tensorflow.python.keras.utils.data_utils import Sequence


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

    #train_generator = (pair for pair in zip(image_generator, mask_generator))
    train_generator = combine_generator(image_generator,mask_generator)
    yield train_generator


# Test generator

def testGenerator(test_path, num_image=2, flag_multi_class=False, target_img_size=config.target_img_size, as_gray=True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path, "%d.png" % i), as_gray=as_gray)
        img = img / 255
        img = trans.resize(img, target_img_size)
        img = np.reshape(img, img.shape + (1,)) if (not flag_multi_class) else img
        img = np.reshape(img, (1,) + img.shape)
        yield img

def combine_generator(gen1, gen2):
    while True:
        yield(next(gen1), next(gen2))