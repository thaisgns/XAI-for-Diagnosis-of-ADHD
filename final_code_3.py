# Import packages

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as skio
import csv
import random
import os

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras import backend as K
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, InputLayer, Lambda, Concatenate
from tensorflow.keras.layers import Conv3D, Conv2D, Convolution2D, Conv2DTranspose, DepthwiseConv2D
from tensorflow.keras.layers import MaxPooling3D, MaxPool2D, MaxPooling2D, Dense, GlobalMaxPooling2D
from tensorflow.keras.layers import Reshape, Flatten, BatchNormalization , Dropout, SpatialDropout2D
from tensorflow.keras.layers import Activation, LeakyReLU
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.metrics import Recall, Precision


import sklearn
from sklearn.metrics import confusion_matrix, roc_auc_score,roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Import packages for the LRP analysis
import innvestigate
import innvestigate.utils as iutils
from innvestigate.analyzer import LRP

# Import packages for the SHAP analysis
import shap

# AAL ROIs

AAL_ROI=["Precentral_L",
"Precentral_R",
"Frontal_Sup_L",
"Frontal_Sup_R",
"Frontal_Sup_Orb_L",
"Frontal_Sup_Orb_R",
"Frontal_Mid_L",
"Frontal_Mid_R",
"Frontal_Mid_Orb_L",
"Frontal_Mid_Orb_R",
"Frontal_Inf_Oper_L",
"Frontal_Inf_Oper_R",
"Frontal_Inf_Tri_L",
"Frontal_Inf_Tri_R",
"Frontal_Inf_Orb_L",
"Frontal_Inf_Orb_R",
"Rolandic_Oper_L",
"Rolandic_Oper_R",
"Supp_Motor_Area_L",
"Supp_Motor_Area_R",
"Olfactory_L",
"Olfactory_R",
"Frontal_Sup_Medial_L",
"Frontal_Sup_Medial_R",
"Frontal_Med_Orb_L",
"Frontal_Med_Orb_R",
"Rectus_L",
"Rectus_R",
"Insula_L",
"Insula_R",
"Cingulum_Ant_L",
"Cingulum_Ant_R",
"Cingulum_Mid_L",
"Cingulum_Mid_R",
"Cingulum_Post_L",
"Cingulum_Post_R",
"Hippocampus_L",
"Hippocampus_R",
"ParaHippocampal_L",
"ParaHippocampal_R",
"Amygdala_L",
"Amygdala_R",
"Calcarine_L",
"Calcarine_R",
"Cuneus_L",
"Cuneus_R",
"Lingual_L",
"Lingual_R",
"Occipital_Sup_L",
"Occipital_Sup_R",
"Occipital_Mid_L",
"Occipital_Mid_R",
"Occipital_Inf_L",
"Occipital_Inf_R",
"Fusiform_L",
"Fusiform_R",
"Postcentral_L",
"Postcentral_R",
"Parietal_Sup_L",
"Parietal_Sup_R",
"Parietal_Inf_L",
"Parietal_Inf_R",
"SupraMarginal_L",
"SupraMarginal_R",
"Angular_L",
"Angular_R",
"Precuneus_L",
"Precuneus_R",
"Paracentral_Lobule_L",
"Paracentral_Lobule_R",
"Caudate_L",
"Caudate_R",
"Putamen_L",
"Putamen_R",
"Pallidum_L",
"Pallidum_R",
"Thalamus_L",
"Thalamus_R",
"Heschl_L",
"Heschl_R",
"Temporal_Sup_L",
"Temporal_Sup_R",
"Temporal_Pole_Sup_L",
"Temporal_Pole_Sup_R",
"Temporal_Mid_L",
"Temporal_Mid_R",
"Temporal_Pole_Mid_L",
"Temporal_Pole_Mid_R",
"Temporal_Inf_L",
"Temporal_Inf_R",
"Cerebelum_Crus1_L",
"Cerebelum_Crus1_R",
"Cerebelum_Crus2_L",
"Cerebelum_Crus2_R",
"Cerebelum_3_L",
"Cerebelum_3_R",
"Cerebelum_4_5_L",
"Cerebelum_4_5_R",
"Cerebelum_6_L",
"Cerebelum_6_R",
"Cerebelum_7b_L",
"Cerebelum_7b_R",
"Cerebelum_8_L",
"Cerebelum_8_R",
"Cerebelum_9_L",
"Cerebelum_9_R",
"Cerebelum_10_L",
"Cerebelum_10_R",
"Vermis_1_2",
"Vermis_3",
"Vermis_4_5",
"Vermis_6",
"Vermis_7",
"Vermis_8",
"Vermis_9",
"Vermis_10"]

# Loading the data for the BCorrU metric
filename='FC_BCorrU.mat'
data=skio.loadmat('FC_BCorrU.mat')

del data['__header__']
del data['__version__']
del data['__globals__']

data=np.array(list(data.values())) #create an array
print('Data dimensions before reshaping',data.shape)
#data=data.reshape(116,116,768) #reshape into 3 dimensions [N_ROIs,N_ROIs,N_subjects]
data = np.moveaxis(data, -1, 0) #reshape into [N_subjects,N_ROIs,N_ROIs]
data=np.moveaxis(data,1,-1)
print('Data dimensions after reshaping',data.shape)

# Loading the labels
filename='Labels_All_Groups.mat'
labels=skio.loadmat(filename)

del labels['__header__']
del labels['__version__']
del labels['__globals__']

lab= np.array(list(labels.values())) #create an array
print("Labels dimensions before reshaping",lab.shape)
lab=lab.reshape(768,1) #reshape into [N_subject, 1]
print("Labels dimensions after reshaping",lab.shape)

total_subjects = len(lab)
n_patients = sum(lab == 1)[0]
n_hc = sum(lab == 0)[0]
id_subjects = np.array(list(range(total_subjects)))
print("From the total of {} subjects, there are {} patients and {} HC ".format(total_subjects, n_patients, n_hc),"\n")

# Displaying a random matrix
fig = plt.figure()
img = plt.imshow(data[401,:,:,0],cmap='seismic')
plt.xlabel('116 AAL brain regions')
plt.ylabel('116 AAL brain regions')
plt.title('FC matrix of a subject with ADHD \n')
plt.colorbar()
plt.show()
print(lab[401][0])

# Checking the symetry of the FC matrices and computing triangular matrices

def triangular_mat(mat) :
  total_subjects = mat.shape[0]
  for i in range (total_subjects) :
      symmetry = np.all(mat[i,:,:,0].transpose() == mat[i,:,:,0])
      if symmetry==True:
        #print("The FC matrix of subject {} is symmetric".format(i))
        # Building the upper triangular matix
        mat[i,:,:,0] = mat[i,:,:,0] * np.tril(np.ones(mat[i,:,:,0].shape)).astype(bool)
        # Putting zeros on the diagonal
        np.fill_diagonal(mat[i,:,:,0], 0)
  return mat

data = triangular_mat(data)
print(data.shape)
# Displaying the triangular matrix
fig = plt.figure()
img = plt.imshow(data[401,:,:,0],cmap='seismic')
plt.xlabel('116 AAL brain regions')
plt.ylabel('116 AAL brain regions')
plt.title('Lower triangular FC matrix of a subject with ADHD \n')
plt.colorbar()
plt.show()

# Defining the model

def ConnectomeCNN(input_shape, drop_pr=0.7, n_filter=32, n_dense1=64, n_classes=2):
    bias_init = tf.constant_initializer(value=0.001)
    input_1 = InputLayer(input_shape=input_shape, name="input")
    conv1 = Conv2D(
        filters=n_filter,
        kernel_size=(1, input_shape[1]),
        strides=(1, 1),
        padding="valid",
        activation="selu",
        kernel_initializer="glorot_uniform",
        #kernel_regularizer=keras.regularizers.l1_l2(l1=0.0001, l2=0.0001), #0.0005
        bias_initializer=bias_init,
        name="conv1",
        input_shape=input_shape
    )

    dropout1 = SpatialDropout2D(drop_pr, name="dropout1")


    conv2 = Conv2D(
        filters=n_filter * 2,
        kernel_size=(input_shape[1], 1),
        strides=(1, 1),
        padding="valid",
        activation="selu",
        kernel_initializer="glorot_uniform",
        #kernel_regularizer=keras.regularizers.l1_l2(l1=0.0001, l2=0.0001), #0.0005
        bias_initializer=bias_init,
        name="conv2"
    )
    dropout2 = Dropout(drop_pr, name="dropout2")


    reshape = Reshape((n_filter * 2,), name="reshape")

    dense1 = Dense(
        (n_dense1 * 2), activation="selu", name="dense1", kernel_regularizer=keras.regularizers.l1_l2() #'l1_l2' 0.005 keras.regularizers.l1_l2()

    )  #kernel_regularizer = regularizers.l1(0.0001)) #keras.regularizers.l1_l2(0.0001)
    dropout4 = Dropout(0.5, name="dropout4")

    activation = "softmax"
    output = Dense(n_classes, activation=activation, name="output")

    model = keras.models.Sequential(
        [input_1, conv1, dropout1, conv2, dropout2, reshape, dense1, dropout4, output] #dropout1  dropout2
    )
    return model

#Custom model evaluation metrics

def new_recall(y_true, y_pred):
    y_true2 =tf.cast(tf.argmax(y_true, axis=1),tf.float32)
    y_pred2=tf.cast(tf.argmax(y_pred, axis=1),tf.float32)
    true_positives = K.sum(K.round(K.clip(y_true2 * y_pred2, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true2, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def new_specificity(y_true,y_pred):
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    false_positives = K.sum(neg_y_true * y_pred)
    true_negatives = K.sum(neg_y_true * neg_y_pred)
    specificity = true_negatives/(true_negatives + false_positives + K.epsilon())
    return specificity

def new_precision(y_true, y_pred):
    y_true2 =tf.cast(tf.argmax(y_true, axis=1),tf.float32)
    y_pred2=tf.cast(tf.argmax(y_pred, axis=1),tf.float32)
    true_positives = K.sum(K.round(K.clip(y_true2 * y_pred2, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred2, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

#def new_NPV(y_true, y_pred):
    #TN, FP, FN, TP =sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
    #npv = TN / (TN + FN + K.epsilon())
    #return npv

def new_f1(y_true, y_pred):
    precision = new_precision(y_true, y_pred)
    recall = new_recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def new_auc(y_true_oneHot, y_pred_oneHot):
  y_true = np.argmax(y_true_oneHot,axis=1)
  y_pred = y_pred_oneHot[:,1]
  roc_val =sklearn.metrics.roc_auc_score(y_true, y_pred)
  return roc_val

def new_confusion_matrix(y_true_oneHot, y_pred_oneHot):
    y_true = np.argmax(y_true_oneHot, axis=1)
    y_pred=np.argmax(y_pred_oneHot, axis=1)
    TN, FP, FN, TP = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
    cm = np.array([[TP,FP],[FN,TN]])
    return cm

# Hyperparmeters
learning_rate = 0.0001
batch_size = 128
#print(batch_size)
n_epochs = 500
n_classes = 2
loss='binary_crossentropy'

input_shape=(116,116,1)

# K-cross validation

n_kFold=10
skf=StratifiedKFold(n_splits=n_kFold,shuffle = True,random_state=42)
train_loss_cv=[]
train_accuracy_cv=[]
val_loss_cv=[]
val_accuracy_cv=[]
val_recall_cv=[]
val_precision_cv=[]
val_f1_cv=[]
val_auc_cv=[]

accuracy_max = 0

for k, (id_train, id_val) in enumerate(skf.split(data, lab)):
  print('\n FOLD',k+1)
  trainData = data[id_train,:,:]
  valData = data[id_val,:,:]
  print(trainData.shape,valData.shape)
  trainLabels = lab[id_train]
  valLabels = lab[id_val]

  input_shape=(116,116,1)
  trainLabels_oneHot = to_categorical(trainLabels)
  valLabels_oneHot=to_categorical(valLabels)
  B=ConnectomeCNN(input_shape, 0.75,42,80,n_classes)
  B.compile(loss=loss, optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate),
            metrics=['accuracy', new_recall,new_precision,new_specificity,new_f1])

  history = B.fit(trainData, trainLabels_oneHot,epochs=n_epochs,batch_size=batch_size)
  #Getting the evalutaion metrics of the train dataset
  train_loss = history.history['loss']
  train_accuracy = history.history['accuracy']

  train_loss_mean_epoch=np.mean(train_loss) #average of the train accuracy on all the epochs
  train_accuracy_mean_epoch=np.mean(train_accuracy) #average of the train loss on all the epocs
  print('Mean train loss :',train_loss_mean_epoch)
  print('Mean train accracy :',train_accuracy_mean_epoch)

  train_loss_cv.append(train_loss_mean_epoch) #list of the train loss mean
  train_accuracy_cv.append(train_accuracy_mean_epoch) #lis of the train accuracy mean

  # Printing the loss and accuracy as a function of the number of epochs
  x=[i for i in range (1,n_epochs+1)]
  plt.figure()
  plt.plot(x,train_loss,'g',label='Train loss')
  plt.title('Train loss as a function of the epochs',fontsize=12)
  plt.legend()
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  #plt.ylim(0,15)
  plt.xlim(1,n_epochs+1)
  plt.show()

  plt.figure()
  plt.plot(x,train_accuracy,'orange')
  plt.title('Train accuracy as a function of the epochs',fontsize=12)
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  #plt.legend()
  plt.ylim(0,1)
  plt.xlim(1,n_epochs+1)
  plt.show()

  pred_data = B.predict(valData)

  # Plotting the ROC curve
  y_true = np.argmax(valLabels_oneHot,axis=1)
  y_pred=pred_data[:,1]

  FPR, TPR, thresholds = roc_curve(y_true, y_pred)
  plt.plot(FPR,TPR,'m')
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver Operating Characteristic (ROC) Curve')
  plt.show()

  #Computing the evaluation metrics on the validation dataset
  print("Evaluate on test data \n")
  scores = B.evaluate(valData, valLabels_oneHot, verbose=0)
  print('Test loss : ', scores[0])
  val_loss_cv.append(scores[0]) #list of the validation loss
  print('Test accuracy :',scores[1])
  val_accuracy_cv.append(scores[1]) #list of the validation accuracy
  print('Recall:',scores[2])
  val_recall_cv.append(scores[2]) #list of the validation recall
  print('Precision:',scores[3])
  val_precision_cv.append(scores[3]) #list of the validation precision
  print('Specificity:',scores[4])
  val_f1_cv.append(scores[5]) #list of the validation f1
  print('F1 score:',scores[5])

  AUC=new_auc(valLabels_oneHot,pred_data) #computing the AUC
  val_auc_cv.append(AUC) #list of the validation AUC
  print('AUC score:',AUC)

  #Computing and displaying the confusion matrix
  print('Confusion Matrix')
  matrix = new_confusion_matrix(valLabels_oneHot, pred_data)
  print(matrix)

  #Selction of the best prediction for the SHAP and LRP analysis
  if (val_accuracy_cv[k]>accuracy_max) :
    print("best fold nb =",k)
    print(val_accuracy_cv[k])
    accuracy_max=val_accuracy_cv[k]
    XAI_valData=valData
    XAI_valLabels=valLabels
    XAI_trainData=trainData
    XAI_trainLabels=trainLabels
    #Removing the softmax layer for futur LRP analysis
    model_without_softmax = innvestigate.model_wo_softmax(B)
    bestB=B
    best_pred=pred_data

#Plotting the evaluation results

x=[i for i in range (1,n_kFold+1)]
plt.figure()
plt.plot(x,val_loss_cv,'g',label='Train loss')
plt.title('Validation loss as a function of the k-folds',fontsize=12)
plt.legend()
plt.xlabel('Fold')
plt.ylabel('Loss')
plt.xlim(1,n_kFold+1)
plt.show()

plt.figure()
plt.plot(x,val_accuracy_cv,'g',label='Train accuracy')
plt.title('Validation accuracy as a function of the k-folds',fontsize=12)
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.legend()
plt.ylim(0,1)
plt.xlim(1,n_kFold+1)
plt.show()

#Printting the average of the evaluation metrics
train_loss_mean = np.mean(train_loss_cv)
train_accuracy_mean=np.mean(train_accuracy_cv)
print("Mean train loss on the k-fold:",train_loss_mean)
print("Mean train accuracy on the k-fold:",train_accuracy_mean)

val_loss_mean = np.mean(val_loss_cv)
val_accuracy_mean=np.mean(val_accuracy_cv)
val_recall_mean = np.mean(val_recall_cv)
val_precision_mean=np.mean(val_precision_cv)
val_f1_mean=np.mean(val_f1_cv)
val_auc_mean=np.mean(val_auc_cv)
print("Mean validation loss on the k-fold:",val_loss_mean)
print("Mean validation accuracy on the k-fold:",val_accuracy_mean)
print("Mean validation recall on the k-fold:",val_recall_mean)
print("Mean validation precision on the k-fold:",val_precision_mean)
print("Mean validation f1-score on the k-fold:",val_f1_mean)
print("Mean validation AUC on the k-fold:",val_auc_mean)

# Selection of the TP an TN for the global analysis

TPid=[]
FPid=[]
TNid=[]
FNid=[]

pred_lab=np.argmax(best_pred,axis=1)

for i in range (XAI_valLabels.shape[0]):
  if XAI_valLabels[i,0]==1:
    if pred_lab[i]==XAI_valLabels[i,0]:
      TPid.append(i)
    else :
      FPid.append(i)
  if XAI_valLabels[i,0]==0 :
    if pred_lab[i]==XAI_valLabels[i][0]:
      TNid.append(i)
    else :
      FNid.append(i)
print(TPid)

# Selecting the index of an ADHD patien for the local analyses
id_local=TPid[0]
print("Class of the patient n°",id_local,
      "selected for the local analysis:",XAI_valLabels[id_local][0])

# Defining usefull functions for XAI analysis interpretation


def get_coordinates(n_max,XAI_analysis):
    sort = np.argsort(XAI_analysis, axis=None)
    flip = np.flip(sort[-n_max:])
    indices = np.divmod(flip,XAI_analysis.shape[1])

    listOfCoordinates = []
    Coordinates = list(zip(indices[0], indices[1]))
    listOfCoordinates.append(Coordinates)
    return listOfCoordinates[0]

def get_ROIs(n_max,LRP_matrix,atlas):
    """ Returns the list of the 2*n_max more relevant brain regions
    (from the n_max pixels with the highest relevance value)
    according to the LRP analysis
    n_max (an interger): the number of point with the higest
    relevance value to consider
    LRP_matrix (an array): the LRP analysis result
    atlas (a list): the atlas used for the computation of
    the FC matrix and LRP_matrix  """
    listOfCoordinates=get_coordinates(n_max,LRP_matrix)
    ROI_name=[]
    for i in range(n_max):

        coord=listOfCoordinates[i]
        x=coord[0]
        y=coord[1]
        ROI_name.append(AAL_ROI[x])
        ROI_name.append(AAL_ROI[y])

    return ROI_name

def get_nb_influencing_pix (FC_matrix,XAI_analysis, model) :
  n=1
  coord = get_coordinates(n,XAI_analysis)
  new_data = np.copy(FC_matrix)
  new_data = np.expand_dims(new_data, axis=0)
  #new_data = np.expand_dims(new_data, axis=3)
  for pt in coord :
    new_data[0,pt[0],pt[1],0]=0
  new_pred = model.predict(new_data)
  print(new_pred)
  while new_pred[0][0]<new_pred[0][1] :
    n +=1
    coord = get_coordinates(n,XAI_analysis)
    for pt in coord :
      new_data[0,pt[0],pt[1],0]=0
    new_pred = model.predict(new_data)
    print(new_pred)
  return n

def get_nbmean_influencing_pix(FC_matrices,XAI_analyses,model) :
  n_list = []
  print(FC_matrices.shape[0])
  for i in range (FC_matrices.shape[0]) :
    print(FC_matrices[i,:,:,:].shape)
    n=get_nb_influencing_pix(FC_matrices[i,:,:,:],XAI_analyses[i],model)
    print(FC_matrices[i,:,:,:].shape)
    print(n)
    n_list.append(n)
  n_mean=np.int(np.around(np.mean(n_list)))
  return n_mean

def AOPC_calculation (L,data,explanation,model,prediction):

  sum = [0 for i in range (data.shape[0])]
  new_data = data

  for i in range (data.shape[0]): # Loop on all the FC matrices from the dataset
    print("\n------- FC matrix ", i)
    # Getting the original prediction for the unperturbated data
    data_i = np.expand_dims(data[i,:,:,:], axis=0)
    original_pred = prediction[i][1]


    # Getting the coordination of the L most relevant pixels (L being the number o)
    coord=get_coordinates(L,explanation[i,:,:,:])


    sum[i]=original_pred-model.predict(data_i)[0][1]

    for l in range (L): # Loop on the number of pixels to modify
      new_data[i,coord[l][0],coord[l][1],:] = 0
      new_pred = model.predict(np.expand_dims(new_data[i,:,:,:],axis=0))[0][1]
      sum[i] += original_pred - new_pred

  # Computing the average over all the FC matrices in the data set
  mean = np.mean(sum)
  # Computing the AOPC value for the data with L pixels perturbated
  return (1/(L+1))*mean

def AOPC_curve(L,data,explanation,model,prediction):
  AOPC_list=[]
  # Case l=0 ie no pixels perturbated
  AOPC_list.append(0.0)
  # Loop on the number of iteration (L is the total number of iteration)
  for l in range (1,L) :
    print("\n ITERATION",l)
    AOPC_list.append(AOPC_calculation(l,data,explanation,model,prediction))
  return AOPC_list

# LRP analysis

# Creating the analyzer
LRP_analyzer_epsilon =
innvestigate.create_analyzer ('lrp.epsilon',model_without_softmax,neuron_selection_mode="index")
#Appling the analyzer
OutputNeuron_index = 1
X_test_LRP=XAI_valData
analysis = LRP_analyzer_epsilon.analyze(X_test_LRP, OutputNeuron_index)

# LRP
# Local explaination

analysis_local=analysis[id_local,:,:,0]
mat_disp=XAI_valData[id_local,:,:,0]
print("Displayed patient class:",XAI_valLabels[id_local,0])

fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15, 7))
im1 = ax1.imshow(analysis_local, cmap='seismic', interpolation='nearest')
ax1.set_title('Heatmap of LRP-epsilon analyis',fontsize=20)
fig.colorbar(im1, ax = ax1)
im2 = ax2.imshow(mat_disp, cmap='seismic', interpolation='nearest')
ax2.set_title('FC matrix for a TP prediction (ADHD patient)',fontsize=15)
fig.colorbar(im2, ax = ax2)

# Testing correctness LRP
L = 100
AOPCdata = XAI_valData[TPid]
AOPCanalysis = analysis[TPid]
AOPCpred=best_pred[TPid]
AOPC_values = AOPC_curve(L,AOPCdata,AOPCanalysis,bestB,AOPCpred)

#Identifing the most relevant brain regions for classification
ROIs_local=get_ROIs(5,analysis_local,AAL_ROI)

print("LOCAL ANALYSIS \nName of the ",10, "most relevant brain regions: ")
for i in range(len(ROIs_local)):
    print(ROIs_local[i])

# LRP
# Global explaination

# average result for LRP from all subjects
analysis_TP=analysis[TPid]
data_TP=XAI_valData[TPid]
analysis_mean = np.mean(analysis_TP, axis = 0)
# average result for FC matrices from subjects in Test Set(X_test)
FC_mat_mean = np.mean(data_TP, axis = 0)

# Displaing the result of the LRP analyzer on the average of the subjects
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15, 7))
im1 = ax1.imshow(analysis_mean, cmap='seismic', interpolation='nearest')
ax1.set_title('Average of the LRP-epsilon heatmaps ',fontsize=15)
fig.colorbar(im1, ax = ax1)
im2 = ax2.imshow(FC_mat_mean, cmap='seismic', interpolation='nearest')
ax2.set_title('Average of all TP FC matrices',fontsize=15)
fig.colorbar(im2, ax = ax2)

ROIs_global=get_ROIs(5,analysis_mean,AAL_ROI)

print("GLOBAL ANALYSIS \nName of the",10,"most relevant brain regions:\n")
for i in range(len(ROIs_global)):
  print(ROIs_global[i])

# SHAP analysis

np.bool=np.bool_
backgroundData = XAI_trainData[np.random.choice(XAI_trainData.shape[0], 500 , replace=False)]
explainer = shap.DeepExplainer(bestB,backgroundData)

# SHAP
# Local explaination

image = XAI_valData[id_local]
image = image.reshape(1,116,116,1)
shap_local=explainer.explainer.shap_values(image, check_additivity=False)
print(shap_local[1].shape)
plt.imshow(shap_local[1][0,:,:,0], cmap='seismic', interpolation='nearest')
plt.title('Heatmap of the SHAP local analysis',fontsize=15)
plt.colorbar()
plt.show()

# Plot using the tools from the SHAP library
shap.image_plot(shap_local[1],image)
print(shap_local[1].shape)

# Most important brain regions
ROIs_local_shap=get_ROIs(5,shap_local[1],AAL_ROI)

print("LOCAL ANALYSIS \nName of the ",10, "most relevant brain regions: ")
for i in range(len(ROIs_local_shap)):
    print(ROIs_local_shap[i])

# SHAP
# Global explaination

shap_values=explainer.explainer.shap_values(XAI_valData, check_additivity=False)
print(shap_values[1].shape)

analysis_mean_SHAP=np.mean(shap_values[1],axis=0) #analysis with respect to the positive class
FC_mat_mean = np.mean(XAI_valData[:,:,:,0], axis = 0)

# Displaying the result of the analysis
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15, 7))
im1 = ax1.imshow(analysis_mean[:,:], cmap='seismic', interpolation='nearest')
ax1.set_title('SHAP values for metric BCorrU',fontsize=20)
fig.colorbar(im1, ax = ax1)
im2 = ax2.imshow(FC_mat_mean, cmap='seismic', interpolation='nearest')
ax2.set_title('FC matrix for metric BCorrU',fontsize=20)
fig.colorbar(im2, ax = ax2)

# Most important brain regions
ROIs_global_shap=get_ROIs(5,analysis_mean_SHAP,AAL_ROI)

print("GLOBAL ANALYSIS \nName of the",10,"most relevant brain regions:\n")
for i in range(len(ROIs_global_shap)):
  print(ROIs_global_shap[i])

# Testing correctness SHAP

L = 100
AOPCdata = XAI_valData[TPid]
AOPCanalysis_shap = shap_values[1][TPid]
AOPCpred=best_pred[TPid]
AOPC_values_shap = AOPC_curve(L,AOPCdata,AOPCanalysis_shap,bestB,AOPCpred)

#Plotting AOPC curves
x=[i for i in range (1,L+1)]
plt.plot(x,AOPC_values_shap,label='SHAP')
plt.plot(x,AOPC_values,'m',label='LRP')
plt.legend()
plt.xlabel("Iterations (L)")
plt.ylabel("AOPC")
plt.title("AOPC for LRP and SHAP methods \n as a function of the number of pixels deleted (iterations)")
plt.show()

# Testing continuity

def SENS_calculation (data,LRP_explainer,SHAP_explainer,model) :
  # Adding noise to the data
  noise_std = 0.05
  noise = np.random.normal(loc=0, scale=noise_std, size=(116,116))
  noise[:,:]= noise[:,:] * np.tril(np.ones(noise.shape)).astype(bool)
  np.fill_diagonal(noise, 0)

  noisy_data = data[0,:,:,0]+ noise

  # Expanding the dimension of noisy_data so it fits the input shape of the explainer
  noisy_data=np.expand_dims(noisy_data,axis=0)
  noisy_data=np.expand_dims(noisy_data,axis=3)

  # Checking that the difference in the explanation isn't too big
  pred=model.predict(data)
  print("pred = ",pred)
  pred_noise=model.predict(noisy_data)
  print("pred noisy =",pred_noise)
  print("Diff pred = ",pred_noise-pred)

  diff_data = noisy_data - data

  # Getting the LRP analysis
  OutputNeuron_index = 1
  analysis_noisy_data_LRP=LRP_explainer.analyze(noisy_data, OutputNeuron_index)
  analysis_data_LRP = LRP_explainer.analyze(data, OutputNeuron_index)

  # Computing LRP sensitivity
  diff_LRP = analysis_noisy_data_LRP[0,:,:,0] - analysis_data_LRP[0,:,:,0]
  SENS_LRP = np.linalg.norm(diff_LRP, 'fro')

  # Getting the SHAP analysis
  analysis_noisy_data_SHAP=
  SHAP_explainer.explainer.shap_values(noisy_data, check_additivity=False)
  analysis_data_SHAP =
  SHAP_explainer.explainer.shap_values(data, check_additivity=False)

  # Computing SHAP sensitivity
  diff_SHAP = analysis_noisy_data_SHAP[1][0,:,:,0] - analysis_data_SHAP[1][0,:,:,0]
  SENS_SHAP = np.linalg.norm(diff_SHAP, 'fro')

  return SENS_LRP, SENS_SHAP

SENS_list_LRP = []
SENS_list_SHAP = []

for i in range(data_TP.shape[0]) : # Loop on all the FC matrices
  print("\n Matrix ",i)
  data = np.expand_dims(data_TP[i],axis=0)
  SENS=SENS_calculation(data,LRP_analyzer_epsilon,explainer,bestB)
  SENS_list_LRP.append(SENS[0])
  SENS_list_SHAP.append(SENS[1])

print("Sensitivity list LRP \n",SENS_list_LRP)
print("Sensitivity list SHAP \n",SENS_list_SHAP)
SENS_LRP = np.mean(SENS_list_LRP)
print("Sensitivity LRP = ",SENS_LRP)
SENS_SHAP = np.mean(SENS_list_SHAP)
print("Sensitivity SHAP = ",SENS_SHAP)