import os
os.chdir(os.getcwd())
from Utils import config, inputData
from Model import unet_model
import matplotlib.pyplot as plt


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

myGenerator = inputData.trainGenerator(8,config.target_data_root  ,'images' ,'mask', data_gen_args, config.target_img_size)


#%%
x_batch = next(myGenerator)
for i in range(3):
    image = x_batch[i]
    plt.imshow(image[1,:,:])
    plt.show()

##

#%%
model = unet_model.unet()
model_checkpoint = unet_model.ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit(myGenerator,steps_per_epoch=1,epochs=1,callbacks=[model_checkpoint])