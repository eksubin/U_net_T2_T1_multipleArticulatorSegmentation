#Locations
target_data_root = "Data/Airway/"
airway_msk_loc = "Data/Airway_mask/"

#Image Parameters
target_img_size = (256, 256)
target_input_size = (256,256,1)


#Training Parameters
n_epochs = 50

#parameter Dictionary for data augmentation
data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')