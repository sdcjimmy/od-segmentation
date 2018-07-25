import numpy as np
import os
import ConfigParser
from matplotlib import pyplot as plt
#Keras
from keras.models import model_from_json
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.utils.vis_utils import plot_model as plot
from keras.optimizers import SGD
from keras.optimizers import Adagrad

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
from model import *
from dual_IDG import *


#Train parameters
experiment= "Segmenter4"
name_experiment= "DISC_segmentation/" + experiment
localizer_weight = "LocUnet_name2"
#training settings
N_epochs = 200
batch_size = 8
patch_height = 480
patch_width = 480
N_subimgs = 73*6*1000
optimizer = "sgd"
learningrate = 0.05
inside_FOV = False
to_extract_patches = False
trim = True
augmentation = True

directory = name_experiment
if not os.path.exists(directory):
    os.makedirs(directory)

with open(name_experiment + "/configuration.txt",'w') as out:
    line0 = '[training settings]'
    line1 = 'N_epochs = ' + str(N_epochs)
    line2 = 'batch_size = ' + str(batch_size)
    line3 = 'patch_height = ' + str(patch_height)
    line4 = 'patch_width = ' + str(patch_width)
    line5 = 'N_subimgs = ' + str(N_subimgs)
    line6 = 'inside_FOV = ' + str(inside_FOV)
    line7 = 'optimizer = ' + optimizer
    line8 = 'learning rate = ' + str(learningrate)
    line9 = 'to_extract_patches = ' + str(to_extract_patches)
    line10 = 'trim = ' + str(trim)
    out.write('{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n'.format(line0,line1,line2,line3,line4,line5,line6, line7, line8, line9, line10)) 
    


# Load data
path_data = "DISC/"

print("loading data ...")
raw_img = load_hdf5("DISC/total_raw.hdf5")
mask_img = load_hdf5("DISC/total_mask.hdf5")

raw_train, raw_test, mask_train, mask_test = train_test_split(raw_img, mask_img, test_size = 0.2, random_state = 123)

if trim == True:
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
    

print("raw train processed shape:" +  str(raw_train_processed.shape))
print("mask train shape:" + str(mask_train.shape))


print("Extracting data ...")
if to_extract_patches == True:
    patches_imgs_train, patches_masks_train = extract_random_with_disc(raw_train_processed,
                                                                   mask_train,
                                                                   patch_height,
                                                                   patch_width,
                                                                   N_subimgs,
                                                                   inside_FOV)
else:
    patches_imgs_train = raw_train_processed
    patches_masks_train = mask_train
    
    
print("Finish extracting !")

#patches_imgs_train, patches_masks_train = extract_random_with_disc(raw_train_processed,mask_train,patch_height,patch_width,N_subimgs,inside_FOV)
#patches_imgs_train = raw_train_processed
#patches_masks_train = mask_train

print(patches_imgs_train.shape)


#=========== Construct and save the model arcitecture =====
n_ch = patches_imgs_train.shape[1]
patch_height = patches_imgs_train.shape[2]
patch_width = patches_imgs_train.shape[3]

model = get_unet_seg2(1,480,480)  #the U-net model
model.load_weights("DISC_segmentation/" + localizer_weight + "/" + localizer_weight + "_best_weights.h5", by_name = True)

adaGrad = Adagrad(lr=learningrate, epsilon=1e-7, decay=1e-6)
sgd = SGD(lr=learningrate, decay=1e-6, momentum=0.9, nesterov=False)
adam = Adam(lr=learningrate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#model.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=['accuracy', mean_IOU_gpu])
model.compile(optimizer=adam, loss=log_dice_loss ,metrics=['accuracy', mean_IOU_gpu, dice_metric])

print "Check: final output of the network:"
print model.output_shape
#plot(model, to_file='./'+name_experiment+'/'+name_experiment + '_model.png')   #check how the model looks like
json_string = model.to_json()
open('./'+name_experiment+'/'+ experiment +'_architecture.json', 'w').write(json_string)


#============  Training ==================================
checkpointer = ModelCheckpoint(filepath='./'+name_experiment+'/'+ experiment +'_best_weights.h5', verbose=1, monitor='val_loss', mode='auto', save_best_only=True) #save at each epoch if the validation decrease


# def step_decay(epoch):
#     lrate = 0.01 #the initial learning rate (by default in keras)
#     if epoch==100:
#         return 0.005
#     else:
#         return lrate
#
# lrate_drop = LearningRateScheduler(step_decay)

#class_weight = {0:1.,1:50.}
#patches_masks_train = masks_Unet(patches_masks_train)  #reduce memory consumption
model.fit(patches_imgs_train, patches_masks_train, nb_epoch=N_epochs, batch_size=batch_size, verbose=2, shuffle=True, validation_split=0.1, callbacks=[checkpointer])


#========== Save and test the last model ===================
model.save_weights('./'+name_experiment+'/'+ experiment +'_last_weights.h5', overwrite=True)




## Testing 
#Python
import numpy as np
import ConfigParser
from matplotlib import pyplot as plt
#Keras
from keras.models import model_from_json
from keras.models import Model
#scikit learn
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
import sys
sys.path.insert(0, './lib/')
# help_functions.py
from help_functions import *
# extract_patches.py
from extract_patches import extract_ordered
from extract_patches import extract_ordered_overlap
from extract_patches import recompone
from extract_patches import recompone_overlap
from extract_patches import paint_border
from extract_patches import kill_border
from extract_patches import pred_only_FOV
from extract_patches import get_data_testing
from extract_patches import get_data_testing_overlap
# pre_processing.py
from pre_processing import my_PreProc

from PIL import Image


# Test Parameters
path_experiment = "DISC_segmentation/"
name_experiment = experiment

input_path = path_experiment+ name_experiment + '/' +  name_experiment 
output_path = path_experiment+  name_experiment +'/' 

config = ConfigParser.RawConfigParser()
config.read(path_experiment + name_experiment + '/' + 'configuration.txt')


best_last = "best"
Imgs_to_test = 15
#patch_height = int(config.get('training settings','patch_height'))
#patch_width = int(config.get('training settings','patch_width'))
patch_height = 480
patch_width = 480
average_mode = False
is_extract = False
trim = True
overlap = True
average_mode = True
stride_h = 2
stride_w = 2

full_img_height = 480
full_img_width = 480
    



test_imgs_original = load_hdf5("DISC/raw_test.hdf5")
test_masks = load_hdf5("DISC/mask_test.hdf5")
#test_imgs_original = load_hdf5(("DRIVE_datasets_training_testing/DRIVE_dataset_imgs_train.hdf5"))
#test_masks = load_hdf5("DRIVE_datasets_training_testing/DRIVE_dataset_groundTruth_train.hdf5")
#test_imgs_original = test_imgs_original[:,:,52:532,42:522]
#test_masks = test_masks[:,:,52:532,42:522]


test_imgs = my_PreProc(test_imgs_original)
#test_masks = test_masks/255.

    #extend both images and masks so they can be divided exactly by the patches dimensions
test_imgs = test_imgs[0:Imgs_to_test,:,:,:]
test_masks = test_masks[0:Imgs_to_test,:,:,:]

if trim == True:
    test_imgs = test_imgs[:,:,:,80:560]
    test_masks = test_masks[:,:,:,80:560]

if is_extract == True:
    test_imgs = paint_border(test_imgs,patch_height,patch_width)
    test_masks = paint_border(test_masks,patch_height,patch_width)
    
    if overlap == True:
        patches_imgs_test = extract_ordered_overlap(test_imgs, patch_height, patch_width,stride_h,stride_w)
        #patches_masks_test = extract_ordered_overlap(test_masks, patch_height, patch_width,stride_h,stride_w)
    else:
        patches_imgs_test = extract_ordered(test_imgs,patch_height,patch_width)
        patches_masks_test = extract_ordered(test_masks,patch_height,patch_width)
else:
    patches_imgs_test = test_imgs
    patches_masks_test = test_masks
    
#model = model_from_json(open(path_experiment+name_experiment +'_architecture.json').read())
model = model_from_json(open(path_experiment+ name_experiment + '/' +  name_experiment + '_architecture.json').read())
model.load_weights(path_experiment + name_experiment + '/' +  name_experiment +  '_'+best_last+'_weights.h5')
#Calculate the predictions
predictions = model.predict(patches_imgs_test, batch_size=32, verbose=2)
print "predicted images size :"
print predictions.shape

#===== Convert the prediction arrays in corresponding images
#pred_imgs = pred_to_imgs(predictions, patch_height, patch_width, "original")
pred_imgs = predictions
orig_imgs = patches_imgs_test
gtruth_masks = patches_masks_test


#if average_mode == True:
#    pred_imgs = recompone_overlap(pred_patches, full_img_height, full_img_width, stride_h, stride_w)# predictions
#    orig_imgs = my_PreProc(test_imgs_original[0:pred_imgs.shape[0],:,:,:])    #originals
#    gtruth_masks = test_masks  #ground truth masks
#else:
#    pred_imgs = recompone(pred_patches,10,10)       # predictions
#    orig_imgs = recompone(patches_imgs_test,10,10)  # originals
#    gtruth_masks = recompone(patches_masks_test,10,10)  #masks
    
# apply the DRIVE masks on the repdictions #set everything outside the FOV to zero!!
#kill_border(pred_imgs, test_border_masks)  #DRIVE MASK  #only for visualization
## back to original dimensions
orig_imgs = orig_imgs[:,:,0:full_img_height,0:full_img_width]
pred_imgs = pred_imgs[:,:,0:full_img_height,0:full_img_width]
gtruth_masks = gtruth_masks[:,:,0:full_img_height,0:full_img_width]
print "Orig imgs shape: " +str(orig_imgs.shape)
print "pred imgs shape: " +str(pred_imgs.shape)
print "Gtruth imgs shape: " +str(gtruth_masks.shape)


N_visual = 5
visualize(group_images(orig_imgs,N_visual),output_path + "all_originals")#.show()
visualize(group_images(pred_imgs,N_visual),output_path + "all_predictions")#.show()
visualize(group_images(gtruth_masks,N_visual),output_path+ "all_groundTruths")#.show()
#visualize results comparing mask and prediction:
assert (orig_imgs.shape[0]==pred_imgs.shape[0] and orig_imgs.shape[0]==gtruth_masks.shape[0])
N_predicted = orig_imgs.shape[0]
group = N_visual
assert (N_predicted%group==0)
for i in range(int(N_predicted/group)):
    orig_stripe = group_images(orig_imgs[i*group:(i*group)+group,:,:,:],group)
    masks_stripe = group_images(gtruth_masks[i*group:(i*group)+group,:,:,:],group)
    pred_stripe = group_images(pred_imgs[i*group:(i*group)+group,:,:,:],group)
    total_img = np.concatenate((orig_stripe,masks_stripe,pred_stripe),axis=0)
    #total_img = np.concatenate((orig_stripe,pred_stripe),axis=0)
    visualize(total_img,output_path + "_Original_GroundTruth_Prediction"+str(i))#.show()
    

for threshold in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
    t = np.where(pred_imgs > threshold, 1, 0)
    
    iou = []
    dice = []

    for i in range(len(pred_imgs)):
        intersect = 0
        union = 0
        true = np.asarray(gtruth_masks[i].reshape(-1))
        scores = np.asarray(t[i].reshape(-1))
        for j in range(len(true)):
            if true[j] == 1 and scores[j] == 1:
                intersect += 1
            if true[j] == 1 or scores[j] == 1:
                union += 1
        
    
    
        iou.append(intersect/float(union))
        dice.append(intersect*2/float(np.sum(true == 1) + np.sum(scores == 1)))

    print("threshold = ", threshold)
    print("mean iou = ", np.mean(iou))
    print("mean dice = ", np.mean(dice))



