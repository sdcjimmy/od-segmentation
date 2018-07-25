import numpy as np
import ConfigParser
from matplotlib import pyplot as plt
#Keras
from keras.models import model_from_json
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Convolution2D, MaxPooling2D, ZeroPadding2D, Input, Embedding, LSTM, merge, Lambda, UpSampling2D, Deconvolution2D, Cropping2D
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.utils.vis_utils import plot_model as plot
from keras.optimizers import SGD
from keras import regularizers

#scikit learn
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import sys
sys.path.insert(0, './lib/')
# help_functions.py
from help_functions import *
# extract_patches.py
from extract_patches import *
from extract_patches import recompone
from extract_patches import recompone_overlap
from extract_patches import paint_border
from extract_patches import kill_border
from extract_patches import pred_only_FOV
from extract_patches import get_data_testing
from extract_patches import get_data_testing_overlap
# pre_processing.py
from help_functions import *
from pre_processing import *
from extract_patches import *
from mask_retina import *
from model import *
from dual_IDG import *


raw_train = load_hdf5("DISC/raw_train.hdf5")
raw_test =  load_hdf5("DISC/raw_test.hdf5")
mask_train = load_hdf5("DISC/mask_train.hdf5")
mask_test = load_hdf5("DISC/mask_test.hdf5")



#Experiment name
experiment= "LocUnet_name6"
name_experiment= "DISC_segmentation/" + experiment
#training settings
N_epochs = 1000
batch_size = 8
optimizer = "Adam"
learningrate = 0.005
trim = True
augmentation = True

directory = name_experiment
if not os.path.exists(directory):
    os.makedirs(directory)

with open(name_experiment + "/configuration.txt",'w') as out:
    line0 = '[training settings]'
    line1 = 'N_epochs = ' + str(N_epochs)
    line2 = 'batch_size = ' + str(batch_size)
    line3 = 'optimizer = ' + optimizer
    line4 = 'learning rate = ' + str(learningrate)
    line5 = 'trim = ' + str(trim)
    out.write('{}\n{}\n{}\n{}\n{}\n{}\n'.format(line0,line1,line2,line3,line4,line5)) 
    
    

def get_unet_loc(n_ch, img_rows=480, img_cols=480):
    inputs = Input((n_ch, img_rows, img_cols))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(inputs)
    conv1 = Dropout(0.3)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv1)
    pool1 = MaxPooling2D((2, 2), data_format='channels_first')(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool1)
    conv2 = Dropout(0.3)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv2)
    pool2 = MaxPooling2D((2, 2), data_format='channels_first')(conv2)

    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool2)
    conv3 = Dropout(0.3)(conv3)
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv3)
    pool3 = MaxPooling2D((2, 2), data_format='channels_first')(conv3)
    
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool3)
    conv4 = Dropout(0.3)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv4)
    pool4 = MaxPooling2D((2, 2), data_format='channels_first')(conv4)

    
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool4)
    conv5 = Dropout(0.3)(conv5)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv5)
    
    conv6= Flatten()(conv5)
    conv6 = Dense(2,activation='relu',kernel_regularizer=regularizers.l2(0.01))(conv6)
    model = Model(input=inputs, output=conv6)
    
    model.compile(optimizer=SGD(lr=0.05, momentum=0.95),loss='mean_squared_error', metrics= ['mse'])

    return model

def get_unet_loc2(n_ch, img_rows=480, img_cols=480):
    inputs = Input((n_ch, img_rows, img_cols))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first', name = "conv1_1")(inputs)
    conv1 = BatchNormalization(axis=1, name = "conv1_2")(conv1)
    conv1 = Dropout(0.5, name = "conv1_3")(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first', name = "conv1_4")(conv1)
    conv1 = BatchNormalization(axis=1, name = "conv1_5")(conv1)
    pool1 = MaxPooling2D((2, 2), data_format='channels_first',name = "conv1_6")(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first', name = "conv2_1")(pool1)
    conv2 = BatchNormalization(axis=1, name = "conv2_2")(conv2)
    conv2 = Dropout(0.5, name = "conv2_3")(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first',name = "conv2_4")(conv2)
    conv2 = BatchNormalization(axis=1,name = "conv2_5")(conv2)
    pool2 = MaxPooling2D((2, 2), data_format='channels_first',name = "conv2_6")(conv2)

    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first',name = "conv3_1")(pool2)
    conv3 = BatchNormalization(axis=1,name = "conv3_2")(conv3)
    conv3 = Dropout(0.5,name = "conv3_3")(conv3)
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first',name = "conv3_4")(conv3)
    conv3 = BatchNormalization(axis=1,name = "conv3_5")(conv3)
    pool3 = MaxPooling2D((2, 2), data_format='channels_first',name = "conv3_6")(conv3)
    
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first',name = "conv4_1")(pool3)
    conv4 = BatchNormalization(axis=1,name = "conv4_2")(conv4)
    conv4 = Dropout(0.5,name = "conv4_3")(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first',name = "conv4_4")(conv4)
    conv4 = BatchNormalization(axis=1,name = "conv4_5")(conv4)
    pool4 = MaxPooling2D((2, 2), data_format='channels_first',name = "conv4_6")(conv4)

    
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first',name = "conv5_1")(pool4)
    conv5 = BatchNormalization(axis=1,name = "conv5_2")(conv5)
    conv5 = Dropout(0.5,name = "conv5_3")(conv5)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first',name = "conv5_4")(conv5)
    conv5 = BatchNormalization(axis=1,name = "conv5_5")(conv5)
    
    conv6= Flatten()(conv5)
    conv6 = Dense(2,activation='relu',kernel_regularizer=regularizers.l2(0.01))(conv6)
    model = Model(input=inputs, output=conv6)
    
    model.compile(optimizer=SGD(lr=0.1, momentum=0.95),loss='mean_squared_error', metrics= ['mse'])


    return model



def get_center(mask):
    center = np.zeros((mask.shape[0],2))
    for i in range(mask.shape[0]):
        disc = np.where(mask[i] == 1)
        x = np.mean(disc[1])
        y = np.mean(disc[2])
        center[i] = [x,y]
    return center

print("loading data ...")
raw_img = load_hdf5("DISC/total_raw.hdf5")
mask_img = load_hdf5("DISC/total_mask.hdf5")

raw_train, raw_test, mask_train, mask_test = train_test_split(raw_img, mask_img, test_size = 0.2, random_state = 123)

raw_train_trimmed = raw_train[:,:,:,80:560]
mask_train_trimmed = mask_train[:,:,:,80:560]
    
print("preprocessing data ...")
raw_train_processed = my_PreProc(raw_train_trimmed)


if augmentation == True:
    train_idg = DualImageDataGenerator(#rescale=1/255.0,
                                   #samplewise_center=True, samplewise_std_normalization=True,
                                   horizontal_flip=True, vertical_flip=True,
                                   rotation_range=50, width_shift_range=0.15, height_shift_range=0.15,
                                   zoom_range=(0.7, 1.3),
                                   fill_mode='constant', cval=0.0)
    print("Images augmentation ...")
    raw_train_aug = raw_train_trimmed
    mask_train_aug = mask_train_trimmed
    for i in range(5):
        x = raw_train_trimmed.swapaxes(1,3)
        y = mask_train_trimmed.swapaxes(1,3)
        new_x, new_y = train_idg.flow(x, y, batch_size=74, shuffle=True).next()
    
        new_x = new_x.swapaxes(1,3)
        new_y = new_y.swapaxes(1,3)
    
        raw_train_aug = np.concatenate((raw_train_aug, new_x), axis = 0)
        mask_train_aug = np.concatenate((mask_train_aug, new_y), axis = 0)
        
    raw_train_processed = my_PreProc(raw_train_aug)
    mask_train = mask_train_aug

    print("Finish augmentation !")
else:
    mask_train = mask_train_trimmed


model = get_unet_loc2(1,480,480)

center_train = get_center(mask_train)


checkpointer = ModelCheckpoint(filepath='./'+name_experiment+'/'+ experiment +'_best_weights.h5', verbose=1, monitor='val_loss', mode='auto', save_best_only=True) #save at each epoch if the validation decrease

sgd = SGD(lr=10e-6)
rmsprop = RMSprop(lr=learningrate, rho=0.9, epsilon=None, decay=0.0)
adam = Adam(lr=learningrate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='mean_squared_error', optimizer=rmsprop, metrics= ['mse'])
model.fit(raw_train_processed, center_train, nb_epoch=N_epochs, batch_size=batch_size, verbose=2, shuffle=True, validation_split=0.1, callbacks=[checkpointer])

