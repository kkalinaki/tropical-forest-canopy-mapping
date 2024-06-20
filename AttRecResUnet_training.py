
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from datetime import datetime 
import cv2
from PIL import Image
from keras import backend, optimizers

import glob
path = "D:/Kassim/canopy"
os.chdir(path)

SIZE = 256 
n_classes = 1
image_dataset = []
name_of_images=[]
for directory_path in glob.glob("D:/Kassim/canopy/images"):
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        name_of_images.append(img_path[19:-4])
        img = cv2.imread(img_path, 1)       
        img = cv2.resize(img, (SIZE, SIZE))
        image_dataset.append(img)
       
#Convert list to array for machine learning processing        
image_dataset = np.array(image_dataset)

#Capture mask/label info as a list
mask_dataset = [] 
for directory_path in glob.glob("D:/Kassim/canopy/masks/"):
    for mask_path in glob.glob(os.path.join(directory_path, "*.png")):
        mask = cv2.imread(mask_path, 0)       
        mask = cv2.resize(mask, (SIZE, SIZE), interpolation = cv2.INTER_NEAREST)  #Otherwise ground truth changes due to interpolation
        mask_dataset.append(mask)
        
#Convert list to array for machine learning processing          
mask_dataset = np.array(mask_dataset)

# from sklearn.preprocessing import LabelEncoder
# labelencoder = LabelEncoder()
# n, h, w = mask_dataset.shape
# mask_dataset_reshaped = mask_dataset.reshape(-1,1)
# mask_dataset_reshaped_encoded = labelencoder.fit_transform(mask_dataset_reshaped)
# mask_dataset_encoded_original_shape = mask_dataset_reshaped_encoded.reshape(n, h, w)

# np.unique(mask_dataset_encoded_original_shape)

# mask_dataset = np.expand_dims(mask_dataset_encoded_original_shape, axis=3)
from keras.utils import normalize
image_dataset = normalize(np.array(image_dataset), axis=1)
#D not normalize masks, just rescale to 0 to 1.
mask_dataset = np.expand_dims((np.array(mask_dataset)),3) /255.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size = 0.15, random_state = 0)

train_name_of_images=name_of_images[0:len(X_train)]
test_name_of_images=name_of_images[len(X_train):]

#Further split training data t a smaller subset for quick testing of models
X_train, X_do_not_use, y_train, y_do_not_use = train_test_split(X_train, y_train, test_size = 0.15, random_state = 0)

print("Class values in the dataset are ... ", np.unique(y_train))  # 0 i

# from tensorflow.keras.utils import to_categorical
# train_masks_cat = to_categorical(y_train, num_classes=n_classes)

# y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))

# test_masks_cat = to_categorical(y_test, num_classes=n_classes)
# y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))


#Sanity check, view few mages
import random
import numpy as np
image_number = random.randint(0, len(X_train))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(np.reshape(X_train[image_number], (256, 256, 3)), cmap='gray')
plt.subplot(122)
plt.imshow(np.reshape(y_train[image_number], (256, 256)), cmap='gray')
plt.show()

#######################################
#Parameters for model

IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH  = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]
num_labels = 1  #multiclass
input_shape = (IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS)
batch_size = 4

#FOCAL LOSS AND DICE METRIC
from focal_loss import BinaryFocalLoss

from models import UNet, Attention_UNet, Attention_RecurrentResUNet, dice_coef, dice_coef_loss, jacard_coef

'''
UNet
'''
unet_model = UNet(input_shape)
unet_model.compile(optimizer=Adam(learning_rate = 1e-4), loss=BinaryFocalLoss(gamma=2), 
              metrics=['accuracy', jacard_coef])


print(unet_model.summary())
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience=8, monitor='val_loss')
start1 = datetime.now() 
unet_history = unet_model.fit(X_train, y_train, 
                    verbose=1,
                    batch_size = batch_size,
                    validation_data=(X_test, y_test ), 
                    shuffle=False,
                    epochs=100)

stop1 = datetime.now()
#Execution time of the model 
execution_time_Unet = stop1-start1
print("UNet execution time is: ", execution_time_Unet)

unet_model.save('binaryclass_unet_100epochs.hdf5')
#____________________________________________
'''
Attention UNet
'''
att_unet_model = Attention_UNet(input_shape)

att_unet_model.compile(optimizer=Adam(learning_rate = 1e-4), loss = BinaryFocalLoss(gamma=2.0), 
             metrics=['accuracy', jacard_coef])


print(att_unet_model.summary())
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience=8, monitor='val_loss')
start2 = datetime.now() 
att_unet_history = att_unet_model.fit(X_train, y_train, 
                    verbose=1,
                    batch_size = batch_size,
                    validation_data=(X_test, y_test ), 
                    shuffle=False,
                    epochs=100)
stop2 = datetime.now()
#Execution time of the model 
execution_time_Att_Unet = stop2-start2
print("Attention UNet execution time is: ", execution_time_Att_Unet)

att_unet_model.save('binaryclass_Attention_UNet_100epochs.hdf5')

#___________________________________________

'''
Attention Recurrent Residual UNet
'''
att_rec_res_unet_model = Attention_RecurrentResUNet(input_shape)

att_rec_res_unet_model.compile(optimizer=Adam(learning_rate = 1e-4), loss = BinaryFocalLoss(gamma=2.0), 
             metrics=['accuracy', jacard_coef])



print(att_rec_res_unet_model.summary())
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience=8, monitor='val_loss')
start3 = datetime.now() 
att_rec_res_unet_history = att_rec_res_unet_model.fit(X_train, y_train, 
                    verbose=1,
                    batch_size = batch_size,
                    validation_data=(X_test, y_test ), 
                    shuffle=False,
                    epochs=100)
stop3 = datetime.now()

#Execution time of the model 
execution_time_Att_Rec_ResUnet = stop3-start3
print("Attention Recurrent ResUnet execution time is: ", execution_time_Att_Rec_ResUnet)

att_rec_res_unet_model.save('binaryclass_AttRecResUnet_100epochs.hdf5')

############################################################################
# convert the history.history dict to a pandas DataFrame and save as csv for
# future plotting
import pandas as pd    
unet_history_df = pd.DataFrame(unet_history.history) 
att_unet_history_df = pd.DataFrame(att_unet_history.history) 
att_rec_res_unet_history_df = pd.DataFrame(att_rec_res_unet_history.history) 

with open('unet_history_df.csv', mode='w') as f:
    unet_history_df.to_csv(f)
    
with open('att_unet_history_df.csv', mode='w') as f:
    att_unet_history_df.to_csv(f)

with open('custom_code_att_res_unet_history_df.csv', mode='w') as f:
    att_rec_res_unet_history_df.to_csv(f)    

#######################################################################
#Check history plots, one model at a time

#Unet history
history1 = unet_history

#plot the training and validation accuracy and loss at each epoch
loss1 = history1.history['loss']
val_loss1 = history1.history['val_loss']
epochs = range(1, len(loss1) + 1)
plt.plot(epochs, loss1, 'y', label='Training loss')
plt.plot(epochs, val_loss1, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc1 = history1.history['dice_coef']
#acc = history.history['accuracy']
val_acc1 = history1.history['val_dice_coef']
#val_acc = history.history['val_accuracy']

plt.plot(epochs, acc1, 'y', label='Training Dice')
plt.plot(epochs, val_acc1, 'r', label='Validation Dice')
plt.title('Training and validation Jacard')
plt.xlabel('Epochs')
plt.ylabel('Dice')
plt.legend()
plt.show()


#attention Unet history
history2 = att_unet_history

loss2 = history2.history['loss']
val_loss2 = history2.history['val_loss']
epochs = range(1, len(loss2) + 1)
plt.plot(epochs, loss2, 'y', label='Training loss')
plt.plot(epochs, val_loss2, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc2 = history2.history['dice_coef']
#acc = history.history['accuracy']
val_acc2 = history2.history['val_dice_coef']
#val_acc = history.history['val_accuracy']

plt.plot(epochs, acc2, 'y', label='Training Dice')
plt.plot(epochs, val_acc2, 'r', label='Validation Dice')
plt.title('Training and validation Jacard')
plt.xlabel('Epochs')
plt.ylabel('Dice')
plt.legend()
plt.show()



#attention resnet unet history
history3 = att_rec_res_unet_history

loss3 = history3.history['loss']
val_loss3 = history3.history['val_loss']
epochs = range(1, len(loss3) + 1)
plt.plot(epochs, loss3, 'y', label='Training loss')
plt.plot(epochs, val_loss3, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc3 = history3.history['dice_coef']
#acc = history.history['accuracy']
val_acc3 = history3.history['val_dice_coef']
#val_acc = history.history['val_accuracy']

plt.plot(epochs, acc3, 'y', label='Training Dice')
plt.plot(epochs, val_acc3, 'r', label='Validation Dice')
plt.title('Training and validation Jacard')
plt.xlabel('Epochs')
plt.ylabel('Dice')
plt.legend()
plt.show()


######################################################
# Combined Loss curves
######################################################

#Combined Training loss curves
plt.plot(epochs, loss1, label=' U-Net Training loss')
# plt.plot(epochs, val_loss1, label='U-net Validation loss')
plt.plot(epochs, loss2, label='Attention U-Net Training loss')
# plt.plot(epochs, val_loss2, label='Attention U-Net Validation loss')
plt.plot(epochs, loss3, label='Attention Residual U-Net Training loss')
# plt.plot(epochs, val_loss3, label='Attention U-Net Residual Validation loss')
plt.title('Combined Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


#Combined Validation Loss curves
# plt.plot(epochs, loss1, label=' U-Net Training loss')
plt.plot(epochs, val_loss1, label='U-net Validation loss')
# plt.plot(epochs, loss2, label='Attention U-Net Training loss')
plt.plot(epochs, val_loss2, label='Attention U-Net Validation loss')
# plt.plot(epochs, loss3, label='Attention Residual U-Net Training loss')
plt.plot(epochs, val_loss3, label='Attention U-Net Residual Validation loss')
plt.title('Combined validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#Combined training and validation loss curves
plt.plot(epochs, loss1, label=' U-Net Training loss')
plt.plot(epochs, val_loss1, label='U-net Validation loss')
plt.plot(epochs, loss2, label='Attention U-Net Training loss')
plt.plot(epochs, val_loss2, label='Attention U-Net Validation loss')
plt.plot(epochs, loss3, label='Attention Residual U-Net Training loss')
plt.plot(epochs, val_loss3, label='Attention U-Net Residual Validation loss')
plt.title('Combined Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()




######################################################
# Combined IoU curves
######################################################

#Combined Training IoU curves
plt.plot(epochs, acc1, label='U-Net Training IoU')
# plt.plot(epochs, val_acc1, label='U-Net Validation IoU')
plt.plot(epochs, acc2, label='Attention U-Net Training IoU')
# plt.plot(epochs, val_acc2, label='Attention U-Net Validation IoU')
plt.plot(epochs, acc3, label='Attention Residual U-Net Training IoU')
# plt.plot(epochs, val_acc3, label='Attention Residual U-Net Validation IoU')
plt.title('Combined Training IoU')
plt.xlabel('Epochs')
plt.ylabel('IoU')
plt.legend()
plt.show()


#Combined Validation IoU curves
# plt.plot(epochs, acc1, label='U-Net Training IoU')
plt.plot(epochs, val_acc1, label='U-Net Validation IoU')
# plt.plot(epochs, acc2, label='Attention U-Net Training IoU')
plt.plot(epochs, val_acc2, label='Attention U-Net Validation IoU')
# plt.plot(epochs, acc3, label='Attention Residual U-Net Training IoU')
plt.plot(epochs, val_acc3, label='Attention Residual U-Net Validation IoU')
plt.title('Combined Validation IoU')
plt.xlabel('Epochs')
plt.ylabel('IoU')
plt.legend()
plt.show()


#Combined Training and validation IoU curves
plt.plot(epochs, acc1, label='U-Net Training IoU')
plt.plot(epochs, val_acc1, label='U-Net Validation IoU')
plt.plot(epochs, acc2, label='Attention U-Net Training IoU')
plt.plot(epochs, val_acc2, label='Attention U-Net Validation IoU')
plt.plot(epochs, acc3, label='Attention Residual U-Net Training IoU')
plt.plot(epochs, val_acc3, label='Attention Residual U-Net Validation IoU')
plt.title('Combined Training and validation IoU')
plt.xlabel('Epochs')
plt.ylabel('IoU')
plt.legend()
plt.show()

#######################################################
#Model 1
#######################################################

model1 = unet_model
model_path1 = "binaryclass_unet_100epochs.hdf5"
model = tf.keras.models.load_model(model_path1, compile=False)

# IoU and Prediction on testing Data for model 1
y_pred_do_not_use=model1.predict(X_do_not_use)
y_pred_do_not_use_argmax=np.argmax(y_pred_do_not_use, axis=3)

from keras.metrics import MeanIoU
n_classes = 2
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(y_do_not_use[:,:,:,0], y_pred_do_not_use_argmax)
print("Mean IoU =", IOU_keras.result().numpy())


confusion_matrix = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)

# Calculating IoU for class 1 and class 2
class1_iou = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[0, 1] + confusion_matrix[1, 0])
class2_iou = confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[1, 0] + confusion_matrix[0, 1])

print("IoU for class 1 is: ", class1_iou)
print("IoU for class 2 is: ", class2_iou)

#plot some images for model 3 using testing data
import random
for i in range(len(X_test)):
    test_img = X_do_not_use[i]
    ground_truth=y_do_not_use[i]
    #test_img_norm=test_img[:,:,0][:,:,None]
    test_img_input=np.expand_dims(test_img, 0)
    prediction = (model1.predict(test_img_input))
    predicted_img=np.argmax(prediction, axis=3)[0,:,:]
    
    plt.figure(figsize=(12, 8))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(test_img[:,:,:], cmap='gray')
    plt.subplot(232)
    plt.title('Testing Label')
    plt.imshow(ground_truth[:,:,0], cmap='jet')
    plt.subplot(233)
    plt.title('Prediction on test image')
    plt.imshow(predicted_img, cmap='jet')
    plt.savefig(r'D:/Kassim/canopy/unet/%s'%test_name_of_images[i][26:]+'.png',dpi=300)
    plt.show()



# IoU and Prediction on Validation Data for model 1
y_pred=model1.predict(X_test)
y_pred_argmax=np.argmax(y_pred, axis=3)

from keras.metrics import MeanIoU
n_classes = 2
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(y_test[:,:,:,0], y_pred_argmax)
print("Mean IoU =", IOU_keras.result().numpy())


confusion_matrix = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)

# Calculating IoU for class 1 and class 2
class1_iou = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[0, 1] + confusion_matrix[1, 0])
class2_iou = confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[1, 0] + confusion_matrix[0, 1])

print("IoU for class 1 is: ", class1_iou)
print("IoU for class 2 is: ", class2_iou)


#plot some images for model 1
# import random
# test_img_number = random.randint(0, len(X_test)-1)
# test_img = X_test[test_img_number]
# ground_truth=y_test[test_img_number]
# #test_img_norm=test_img[:,:,0][:,:,None]
# test_img_input=np.expand_dims(test_img, 0)
# prediction = (model1.predict(test_img_input))
# predicted_img=np.argmax(prediction, axis=3)[0,:,:]


# plt.figure(figsize=(12, 8))
# plt.subplot(231)
# plt.title('Testing Image')
# plt.imshow(test_img[:,:,0], cmap='gray')
# plt.subplot(232)
# plt.title('Testing Label')
# plt.imshow(ground_truth[:,:,0], cmap='jet')
# plt.subplot(233)
# plt.title('Prediction on test image')
# plt.imshow(predicted_img, cmap='jet')
# plt.show()


#######################################################
#Model 2
#######################################################
model2 = att_unet_model
model_path2 = "binaryclass_Attention_UNet_100epochs.hdf5"
model = tf.keras.models.load_model(model_path2, compile=False)

# IoU and Prediction on testing Data for model 2
y_pred_do_not_use=model2.predict(X_do_not_use)
y_pred_do_not_use_argmax=np.argmax(y_pred_do_not_use, axis=3)

from keras.metrics import MeanIoU
n_classes = 2
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(y_do_not_use[:,:,:,0], y_pred_do_not_use_argmax)
print("Mean IoU =", IOU_keras.result().numpy())


confusion_matrix = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)

# Calculating IoU for class 1 and class 2
class1_iou = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[0, 1] + confusion_matrix[1, 0])
class2_iou = confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[1, 0] + confusion_matrix[0, 1])

print("IoU for class 1 is: ", class1_iou)
print("IoU for class 2 is: ", class2_iou)

#plot some images for model 3 using testing data
for i in range(len(X_test)):
    test_img = X_do_not_use[i]
    ground_truth=y_do_not_use[i]
    #test_img_norm=test_img[:,:,0][:,:,None]
    test_img_input=np.expand_dims(test_img, 0)
    prediction = (model2.predict(test_img_input))
    predicted_img=np.argmax(prediction, axis=3)[0,:,:]
    
    
    plt.figure(figsize=(12, 8))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(test_img[:,:,:], cmap='gray')
    plt.subplot(232)
    plt.title('Testing Label')
    plt.imshow(ground_truth[:,:,0], cmap='jet')
    plt.subplot(233)
    plt.title('Prediction on test image')
    plt.imshow(predicted_img, cmap='jet')
    plt.savefig(r'D:/Kassim/canopy/attention/%s'%test_name_of_images[i][26:]+'.png',dpi=300)
    plt.show()

# IoU and Prediction on Validation Data for model 2
y_pred=model2.predict(X_test)
y_pred_argmax=np.argmax(y_pred, axis=3)

from keras.metrics import MeanIoU
n_classes = 2
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(y_test[:,:,:,0], y_pred_argmax)
print("Mean IoU =", IOU_keras.result().numpy())


confusion_matrix = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)

# Calculating IoU for class 1 and class 2
class1_iou = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[0, 1] + confusion_matrix[1, 0])
class2_iou = confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[1, 0] + confusion_matrix[0, 1])

print("IoU for class 1 is: ", class1_iou)
print("IoU for class 2 is: ", class2_iou)


#plot some images for model 2
# import random
# test_img_number = random.randint(0, len(X_test)-1)
# test_img = X_test[test_img_number]
# ground_truth=y_test[test_img_number]
# #test_img_norm=test_img[:,:,0][:,:,None]
# test_img_input=np.expand_dims(test_img, 0)
# prediction = (model2.predict(test_img_input))
# predicted_img=np.argmax(prediction, axis=3)[0,:,:]


# plt.figure(figsize=(12, 8))
# plt.subplot(231)
# plt.title('Testing Image')
# plt.imshow(test_img[:,:,0], cmap='gray')
# plt.subplot(232)
# plt.title('Testing Label')
# plt.imshow(ground_truth[:,:,0], cmap='jet')
# plt.subplot(233)
# plt.title('Prediction on test image')
# plt.imshow(predicted_img, cmap='jet')
# plt.show()



#######################################################
#Model 3
#######################################################
model3 = att_rec_res_unet_model
model_path3 = "binaryclass_AttRecResUnet_100epochs.hdf5"
model = tf.keras.models.load_model(model_path3, compile=False)

# IoU and Prediction on testing Data for model 3
y_pred_do_not_use=model3.predict(X_do_not_use)
y_pred_do_not_use_argmax=np.argmax(y_pred_do_not_use, axis=3)

from keras.metrics import MeanIoU
n_classes = 2
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(y_do_not_use[:,:,:,0], y_pred_do_not_use_argmax)
print("Mean IoU =", IOU_keras.result().numpy())


confusion_matrix = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)

# Calculating IoU for class 1 and class 2
class1_iou = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[0, 1] + confusion_matrix[1, 0])
class2_iou = confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[1, 0] + confusion_matrix[0, 1])

print("IoU for class 1 is: ", class1_iou)
print("IoU for class 2 is: ", class2_iou)

#plot some images for model 3 using testing data
for i in range(len(X_test)):
    test_img = X_do_not_use[i]
    ground_truth=y_do_not_use[i]
    #test_img_norm=test_img[:,:,0][:,:,None]
    test_img_input=np.expand_dims(test_img, 0)
    prediction = (model3.predict(test_img_input))
    predicted_img=np.argmax(prediction, axis=3)[0,:,:]
    
    
    plt.figure(figsize=(12, 8))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(test_img[:,:,:], cmap='gray')
    plt.subplot(232)
    plt.title('Testing Label')
    plt.imshow(ground_truth[:,:,0], cmap='jet')
    plt.subplot(233)
    plt.title('Prediction on test image')
    plt.imshow(predicted_img, cmap='jet')
    plt.savefig(r'D:/Kassim/canopy/att_rec_res_unet/%s'%test_name_of_images[i][26:]+'.png',dpi=300)
    plt.show()




#### IoU and Prediction on Validation Data for model 3 ###
y_pred=model3.predict(X_test)
y_pred_argmax=np.argmax(y_pred, axis=3)

from keras.metrics import MeanIoU
n_classes = 2
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(y_test[:,:,:,0], y_pred_argmax)
print("Mean IoU =", IOU_keras.result().numpy())


confusion_matrix = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)

# Calculating IoU for class 1 and class 2
class1_iou = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[0, 1] + confusion_matrix[1, 0])
class2_iou = confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[1, 0] + confusion_matrix[0, 1])

print("IoU for class 1 is: ", class1_iou)
print("IoU for class 2 is: ", class2_iou)

#plot some images for model 3 using validation data
# import random
# test_img_number = random.randint(0, len(X_test)-1)
# test_img = X_test[test_img_number]
# ground_truth=y_test[test_img_number]
# #test_img_norm=test_img[:,:,0][:,:,None]
# test_img_input=np.expand_dims(test_img, 0)
# prediction = (model3.predict(test_img_input))
# predicted_img=np.argmax(prediction, axis=3)[0,:,:]


# plt.figure(figsize=(12, 8))
# plt.subplot(231)
# plt.title('Testing Image')
# plt.imshow(test_img[:,:,0], cmap='gray')
# plt.subplot(232)
# plt.title('Testing Label')
# plt.imshow(ground_truth[:,:,0], cmap='jet')
# plt.subplot(233)
# plt.title('Prediction on test image')
# plt.imshow(predicted_img, cmap='jet')
# plt.show()
