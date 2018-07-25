#Keras
from keras.models import model_from_json
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Convolution2D, MaxPooling2D, ZeroPadding2D, Input, Embedding, LSTM, merge, Lambda, UpSampling2D, Deconvolution2D, Cropping2D
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.utils.vis_utils import plot_model as plot
from keras.optimizers import SGD
from keras.optimizers import Adagrad



def mean_IOU_gpu(X, Y):
    """Computes mean Intersection-over-Union (IOU) for two arrays of binary images.
    Assuming X and Y are of shape (n_images, w, h)."""
    
    #X_fl = K.clip(K.batch_flatten(X), K.epsilon(), 1.)
    #Y_fl = K.clip(K.batch_flatten(Y), K.epsilon(), 1.)
    X_fl = K.clip(K.batch_flatten(X), 0., 1.)
    Y_fl = K.clip(K.batch_flatten(Y), 0., 1.)
    X_fl = K.cast(K.greater(X_fl, 0.5), 'float32')
    Y_fl = K.cast(K.greater(Y_fl, 0.5), 'float32')

    intersection = K.sum(X_fl * Y_fl, axis=1)
    union = K.sum(K.maximum(X_fl, Y_fl), axis=1)
    # if union == 0, it follows that intersection == 0 => score should be 0.
    union = K.switch(K.equal(union, 0), K.ones_like(union), union)
    return K.mean(intersection / K.cast(union, 'float32'))


def mean_IOU_gpu_loss(X, Y):
    return -mean_IOU_gpu(X, Y)

def log_mean_IOU_gpu_loss(X, Y):
    return -K.log(mean_IOU_gpu(X, Y))

def dice(y_true, y_pred):
    # Workaround for shape bug. For some reason y_true shape was not being set correctly
    #y_true.set_shape(y_pred.get_shape())

    # Without K.clip, K.sum() behaves differently when compared to np.count_nonzero()
    #y_true_f = K.clip(K.batch_flatten(y_true), K.epsilon(), 1.)
    #y_pred_f = K.clip(K.batch_flatten(y_pred), K.epsilon(), 1.)
    y_true_f = K.clip(K.batch_flatten(y_true), 0., 1.)
    y_pred_f = K.clip(K.batch_flatten(y_pred), 0., 1.)
    #y_pred_f = K.cast(K.greater(y_pred, 0.1), 'float32')

    intersection = 2 * K.sum(y_true_f * y_pred_f, axis=1)
    union = K.sum(y_true_f * y_true_f, axis=1) + K.sum(y_pred_f * y_pred_f, axis=1)
    return K.mean(intersection / union)


def dice_loss(y_true, y_pred):
    return -dice(y_true, y_pred)


def log_dice_loss(y_true, y_pred):
    return -K.log(dice(y_true, y_pred))


def dice_metric(y_true, y_pred):
    """An exact Dice score for binary tensors."""
    y_true_f = K.cast(K.greater(y_true, 0.5), 'float32')
    y_pred_f = K.cast(K.greater(y_pred, 0.5), 'float32')
    return dice(y_true_f, y_pred_f)

def get_unet(n_ch,patch_height,patch_width):
    inputs = Input(shape=(n_ch,patch_height,patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv3)

    up1 = UpSampling2D(size=(2, 2))(conv3)
    up1 = concatenate([conv2,up1],axis=1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv4)
    #
    up2 = UpSampling2D(size=(2, 2))(conv4)
    up2 = concatenate([conv1,up2], axis=1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv5)
    #
    conv6 = Conv2D(2, (1, 1), activation='relu',padding='same',data_format='channels_first')(conv5)
    conv6 = core.Reshape((2,patch_height*patch_width))(conv6)
    conv6 = core.Permute((2,1))(conv6)
    ############
    conv7 = core.Activation('softmax')(conv6)

    model = Model(input=inputs, output=conv7)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    #sgd = SGD(lr = 1e-7, decay = 1e-6, momentum = 0.9, 
    adaGrad = Adagrad(lr=1e-7, epsilon=1e-7, decay=1e-6)
    model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])

    return model

def get_unet_iou(n_ch,patch_height,patch_width):
    inputs = Input(shape=(n_ch,patch_height,patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv3)

    up1 = UpSampling2D(size=(2, 2))(conv3)
    up1 = concatenate([conv2,up1],axis=1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv4)
    #
    up2 = UpSampling2D(size=(2, 2))(conv4)
    up2 = concatenate([conv1,up2], axis=1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv5)
    #
    conv6 = Conv2D(2, (1, 1), activation='relu',padding='same',data_format='channels_first')(conv5)
    conv6 = core.Reshape((2,patch_height*patch_width))(conv6)
    conv6 = core.Permute((2,1))(conv6)
    ############
    conv7 = core.Activation('softmax')(conv6)

    model = Model(input=inputs, output=conv7)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    #sgd = SGD(lr = 1e-7, decay = 1e-6, momentum = 0.9, 
    adaGrad = Adagrad(lr=0.01, epsilon=1e-7, decay=1e-6)
    model.compile(optimizer=adaGrad, loss='categorical_crossentropy',metrics=['accuracy', mean_IOU_gpu])

    return model



def get_unet_light(n_ch, img_rows=480, img_cols=480):
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
    
    up1 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv5)
    up1 = concatenate([conv4,up1],axis=1)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(up1)
    conv6 = Dropout(0.3)(conv6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv6)

    up2 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv6)
    up2 = concatenate([conv3,up2],axis=1)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(up2)
    conv7 = Dropout(0.3)(conv7)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv7)

    up3 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv7)
    up3 = concatenate([conv2,up3],axis=1)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(up3)
    conv8 = Dropout(0.3)(conv8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv8)

    up4 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv8)
    up4 = concatenate([conv1,up4],axis=1)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(up4)
    conv9 = Dropout(0.3)(conv9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv9)
       
    conv10 = Conv2D(1,(1,1), activation='softmax', padding = 'same', data_format='channels_first')(conv9)

   
    model = Model(input=inputs, output=conv10)
    model.compile(optimizer=SGD(lr=0.1, momentum=0.95),loss=log_dice_loss, metrics=[mean_IOU_gpu, dice_metric])

    return model




def get_unet2(n_ch,patch_height,patch_width):
    inputs = Input(shape=(n_ch,patch_height,patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(inputs)
    conv1 = BatchNormalization(axis=1)(conv1)
    conv1 = Dropout(0.3)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv1)
    conv1 = BatchNormalization(axis=1)(conv1)
    pool1 = MaxPooling2D((2, 2), data_format='channels_first')(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool1)
    conv2 = BatchNormalization(axis=1)(conv2)
    conv2 = Dropout(0.3)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv2)
    conv2 = BatchNormalization(axis=1)(conv2)
    pool2 = MaxPooling2D((2, 2), data_format='channels_first')(conv2)

    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool2)
    conv3 = BatchNormalization(axis=1)(conv3)
    conv3 = Dropout(0.3)(conv3)
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv3)
    conv3 = BatchNormalization(axis=1)(conv3)
    pool3 = MaxPooling2D((2, 2), data_format='channels_first')(conv3)
    
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool3)
    conv4 = BatchNormalization(axis=1)(conv4)
    conv4 = Dropout(0.3)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv4)
    conv4 = BatchNormalization(axis=1)(conv4)
    pool4 = MaxPooling2D((2, 2), data_format='channels_first')(conv4)

    
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool4)
    conv5 = BatchNormalization(axis=1)(conv5)
    conv5 = Dropout(0.3)(conv5)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv5)
    conv5 = BatchNormalization(axis=1)(conv5)
    
    up1 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv5)
    up1 = concatenate([conv4,up1],axis=1)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(up1)
    conv6 = BatchNormalization(axis=1)(conv6)
    conv6 = Dropout(0.3)(conv6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv6)
    conv6 = BatchNormalization(axis=1)(conv6)

    up2 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv6)
    up2 = concatenate([conv3,up2],axis=1)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(up2)
    conv7 = BatchNormalization(axis=1)(conv7)
    conv7 = Dropout(0.3)(conv7)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv7)
    conv7 = BatchNormalization(axis=1)(conv7)

    up3 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv7)
    up3 = concatenate([conv2,up3],axis=1)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(up3)
    conv8 = BatchNormalization(axis=1)(conv8)
    conv8 = Dropout(0.3)(conv8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv8)
    conv8 = BatchNormalization(axis=1)(conv8)

    up4 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv8)
    up4 = concatenate([conv1,up4],axis=1)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(up4)
    conv9 = BatchNormalization(axis=1)(conv9)
    conv9 = Dropout(0.3)(conv9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv9)
    conv9 = BatchNormalization(axis=1)(conv9)

    
    conv10 = Conv2D(2, (1, 1), activation='relu', padding='same',data_format='channels_first')(conv9)
    conv10 = BatchNormalization(axis=1)(conv10)
    conv10 = core.Reshape((2,patch_height*patch_width))(conv10)
    conv10 = core.Permute((2,1))(conv10)
    ############
    conv10 = core.Activation('softmax')(conv10)

    model = Model(input=inputs, output=conv10)

    adaGrad = Adagrad(lr=1e-7, epsilon=1e-7, decay=1e-6)
    model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])

    return model


def get_unet_seg(n_ch, img_rows=480, img_cols=480):
    inputs = Input((n_ch, img_rows, img_cols))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first', name = "conv1_1")(inputs)
    conv1 = BatchNormalization(axis=1, name = "conv1_2")(conv1)
    conv1 = Dropout(0.5, name = "conv1_3")(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first', name = "conv1_4")(conv1)
    conv1 = BatchNormalization(axis=1, name = "conv1_5")(conv1)
    conv1.trainable = False
    pool1 = MaxPooling2D((2, 2), data_format='channels_first',name = "conv1_6")(conv1)
    pool1.trainable = False
    
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first', name = "conv2_1")(pool1)
    conv2 = BatchNormalization(axis=1, name = "conv2_2")(conv2)
    conv2 = Dropout(0.5, name = "conv2_3")(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first',name = "conv2_4")(conv2)
    conv2 = BatchNormalization(axis=1,name = "conv2_5")(conv2)
    conv2.trainable = False
    pool2 = MaxPooling2D((2, 2), data_format='channels_first',name = "conv2_6")(conv2)
    pool2.trainable = False
    
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first',name = "conv3_1")(pool2)
    conv3 = BatchNormalization(axis=1,name = "conv3_2")(conv3)
    conv3 = Dropout(0.5,name = "conv3_3")(conv3)
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first',name = "conv3_4")(conv3)
    conv3 = BatchNormalization(axis=1,name = "conv3_5")(conv3)
    conv3.trainable = False
    pool3 = MaxPooling2D((2, 2), data_format='channels_first',name = "conv3_6")(conv3)
    pool3.trainable = False
    
    
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first',name = "conv4_1")(pool3)
    conv4 = BatchNormalization(axis=1,name = "conv4_2")(conv4)
    conv4 = Dropout(0.5,name = "conv4_3")(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first',name = "conv4_4")(conv4)
    conv4 = BatchNormalization(axis=1,name = "conv4_5")(conv4)
    conv4.trainable = False
    pool4 = MaxPooling2D((2, 2), data_format='channels_first',name = "conv4_6")(conv4)
    pool4.trainable = False
    
    
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first',name = "conv5_1")(pool4)
    conv5 = BatchNormalization(axis=1,name = "conv5_2")(conv5)
    conv5 = Dropout(0.5,name = "conv5_3")(conv5)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first',name = "conv5_4")(conv5)
    conv5 = BatchNormalization(axis=1,name = "conv5_5")(conv5)
    conv5.trainable = False
    
    up1 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv5)
    up1 = concatenate([conv4,up1],axis=1)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(up1)
    conv6 = BatchNormalization(axis=1)(conv6)
    conv6 = Dropout(0.3)(conv6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv6)
    conv6 = BatchNormalization(axis=1)(conv6)

    up2 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv6)
    up2 = concatenate([conv3,up2],axis=1)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(up2)
    conv7 = BatchNormalization(axis=1)(conv7)
    conv7 = Dropout(0.3)(conv7)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv7)
    conv7 = BatchNormalization(axis=1)(conv7)

    up3 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv7)
    up3 = concatenate([conv2,up3],axis=1)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(up3)
    conv8 = BatchNormalization(axis=1)(conv8)
    conv8 = Dropout(0.3)(conv8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv8)
    conv8 = BatchNormalization(axis=1)(conv8)

    up4 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv8)
    up4 = concatenate([conv1,up4],axis=1)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(up4)
    conv9 = BatchNormalization(axis=1)(conv9)
    conv9 = Dropout(0.3)(conv9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv9)
    conv9 = BatchNormalization(axis=1)(conv9)

    
    conv10 = Conv2D(2, (1, 1), activation='relu', padding='same',data_format='channels_first')(conv9)
    conv10 = BatchNormalization(axis=1)(conv10)
    conv10 = core.Reshape((2,img_rows*img_cols))(conv10)
    conv10 = core.Permute((2,1))(conv10)
    ############
    conv10 = core.Activation('softmax')(conv10)

    model = Model(input=inputs, output=conv10)

    adaGrad = Adagrad(lr=1e-7, epsilon=1e-7, decay=1e-6)
    model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])

    return model


def get_unet_seg2(n_ch, img_rows=480, img_cols=480):
    inputs = Input((n_ch, img_rows, img_cols))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first', name = "conv1_1")(inputs)
    conv1 = BatchNormalization(axis=1, name = "conv1_2")(conv1)
    conv1 = Dropout(0.5, name = "conv1_3")(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first', name = "conv1_4")(conv1)
    conv1 = BatchNormalization(axis=1, name = "conv1_5")(conv1)
    conv1.trainable = False
    pool1 = MaxPooling2D((2, 2), data_format='channels_first',name = "conv1_6")(conv1)
    pool1.trainable = False
    
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first', name = "conv2_1")(pool1)
    conv2 = BatchNormalization(axis=1, name = "conv2_2")(conv2)
    conv2 = Dropout(0.5, name = "conv2_3")(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first',name = "conv2_4")(conv2)
    conv2 = BatchNormalization(axis=1,name = "conv2_5")(conv2)
    conv2.trainable = False
    pool2 = MaxPooling2D((2, 2), data_format='channels_first',name = "conv2_6")(conv2)
    pool2.trainable = False
    
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first',name = "conv3_1")(pool2)
    conv3 = BatchNormalization(axis=1,name = "conv3_2")(conv3)
    conv3 = Dropout(0.5,name = "conv3_3")(conv3)
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first',name = "conv3_4")(conv3)
    conv3 = BatchNormalization(axis=1,name = "conv3_5")(conv3)
    conv3.trainable = False
    pool3 = MaxPooling2D((2, 2), data_format='channels_first',name = "conv3_6")(conv3)
    pool3.trainable = False
    
    
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first',name = "conv4_1")(pool3)
    conv4 = BatchNormalization(axis=1,name = "conv4_2")(conv4)
    conv4 = Dropout(0.5,name = "conv4_3")(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first',name = "conv4_4")(conv4)
    conv4 = BatchNormalization(axis=1,name = "conv4_5")(conv4)
    conv4.trainable = False
    pool4 = MaxPooling2D((2, 2), data_format='channels_first',name = "conv4_6")(conv4)
    pool4.trainable = False
    
    
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first',name = "conv5_1")(pool4)
    conv5 = BatchNormalization(axis=1,name = "conv5_2")(conv5)
    conv5 = Dropout(0.5,name = "conv5_3")(conv5)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first',name = "conv5_4")(conv5)
    conv5 = BatchNormalization(axis=1,name = "conv5_5")(conv5)
    conv5.trainable = False
    
    up1 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv5)
    up1 = concatenate([conv4,up1],axis=1)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(up1)
    conv6 = BatchNormalization(axis=1)(conv6)
    conv6 = Dropout(0.3)(conv6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv6)
    conv6 = BatchNormalization(axis=1)(conv6)

    up2 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv6)
    up2 = concatenate([conv3,up2],axis=1)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(up2)
    conv7 = BatchNormalization(axis=1)(conv7)
    conv7 = Dropout(0.3)(conv7)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv7)
    conv7 = BatchNormalization(axis=1)(conv7)

    up3 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv7)
    up3 = concatenate([conv2,up3],axis=1)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(up3)
    conv8 = BatchNormalization(axis=1)(conv8)
    conv8 = Dropout(0.3)(conv8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv8)
    conv8 = BatchNormalization(axis=1)(conv8)

    up4 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv8)
    up4 = concatenate([conv1,up4],axis=1)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(up4)
    conv9 = BatchNormalization(axis=1)(conv9)
    conv9 = Dropout(0.3)(conv9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv9)
    conv9 = BatchNormalization(axis=1)(conv9)

    
    conv10 = Conv2D(1, (1, 1), activation='relu', padding='same',data_format='channels_first')(conv9)
    conv10 = BatchNormalization(axis=1)(conv10)
    ############
    conv10 = core.Activation('sigmoid')(conv10)

    model = Model(input=inputs, output=conv10)


    return model


def get_unet_trainable_seg(n_ch, img_rows=480, img_cols=480):
    inputs = Input((n_ch, img_rows, img_cols))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first', name = "conv1_1")(inputs)
    conv1 = BatchNormalization(axis=1, name = "conv1_2")(conv1)
    conv1 = Dropout(0.5, name = "conv1_3")(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first', name = "conv1_4")(conv1)
    conv1 = BatchNormalization(axis=1, name = "conv1_5")(conv1)
    conv1.trainable = True
    pool1 = MaxPooling2D((2, 2), data_format='channels_first',name = "conv1_6")(conv1)
    pool1.trainable = True
    
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first', name = "conv2_1")(pool1)
    conv2 = BatchNormalization(axis=1, name = "conv2_2")(conv2)
    conv2 = Dropout(0.5, name = "conv2_3")(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first',name = "conv2_4")(conv2)
    conv2 = BatchNormalization(axis=1,name = "conv2_5")(conv2)
    conv2.trainable = True
    pool2 = MaxPooling2D((2, 2), data_format='channels_first',name = "conv2_6")(conv2)
    pool2.trainable = True
    
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first',name = "conv3_1")(pool2)
    conv3 = BatchNormalization(axis=1,name = "conv3_2")(conv3)
    conv3 = Dropout(0.5,name = "conv3_3")(conv3)
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first',name = "conv3_4")(conv3)
    conv3 = BatchNormalization(axis=1,name = "conv3_5")(conv3)
    conv3.trainable = True
    pool3 = MaxPooling2D((2, 2), data_format='channels_first',name = "conv3_6")(conv3)
    pool3.trainable = True
    
    
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first',name = "conv4_1")(pool3)
    conv4 = BatchNormalization(axis=1,name = "conv4_2")(conv4)
    conv4 = Dropout(0.5,name = "conv4_3")(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first',name = "conv4_4")(conv4)
    conv4 = BatchNormalization(axis=1,name = "conv4_5")(conv4)
    conv4.trainable = True
    pool4 = MaxPooling2D((2, 2), data_format='channels_first',name = "conv4_6")(conv4)
    pool4.trainable = True
    
    
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first',name = "conv5_1")(pool4)
    conv5 = BatchNormalization(axis=1,name = "conv5_2")(conv5)
    conv5 = Dropout(0.5,name = "conv5_3")(conv5)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first',name = "conv5_4")(conv5)
    conv5 = BatchNormalization(axis=1,name = "conv5_5")(conv5)
    conv5.trainable = True
    
    up1 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv5)
    up1 = concatenate([conv4,up1],axis=1)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(up1)
    conv6 = BatchNormalization(axis=1)(conv6)
    conv6 = Dropout(0.3)(conv6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv6)
    conv6 = BatchNormalization(axis=1)(conv6)

    up2 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv6)
    up2 = concatenate([conv3,up2],axis=1)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(up2)
    conv7 = BatchNormalization(axis=1)(conv7)
    conv7 = Dropout(0.3)(conv7)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv7)
    conv7 = BatchNormalization(axis=1)(conv7)

    up3 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv7)
    up3 = concatenate([conv2,up3],axis=1)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(up3)
    conv8 = BatchNormalization(axis=1)(conv8)
    conv8 = Dropout(0.3)(conv8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv8)
    conv8 = BatchNormalization(axis=1)(conv8)

    up4 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv8)
    up4 = concatenate([conv1,up4],axis=1)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(up4)
    conv9 = BatchNormalization(axis=1)(conv9)
    conv9 = Dropout(0.3)(conv9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv9)
    conv9 = BatchNormalization(axis=1)(conv9)

    
    conv10 = Conv2D(1, (1, 1), activation='relu', padding='same',data_format='channels_first')(conv9)
    conv10 = BatchNormalization(axis=1)(conv10)
    ############
    conv10 = core.Activation('sigmoid')(conv10)

    model = Model(input=inputs, output=conv10)


    return model


