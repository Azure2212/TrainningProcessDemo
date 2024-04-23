import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os 
import random 
import cv2

import keras
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda
from keras import backend as K
from keras.activations import sigmoid
from keras import layers
from keras import initializers
from keras.models import Sequential
import copy
import tensorflow as tf
from tensorflow.keras import Model
import seaborn as sns
from sklearn.metrics import confusion_matrix

from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D,ReLU, Dense, Add
import albumentations as A

##################################################### Model session #####################################################
#1.Attention Define
def cbam_block(cbam_feature, ratio=8, type_resnet=0):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """
    cbam_feature = channel_attention(cbam_feature, type_resnet, ratio)

    cbam_feature = spatial_attention(cbam_feature, type_resnet)
    
    return cbam_feature

def channel_attention(input_feature, type_resnet=0, ratio=8):
    
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]
    
    shared_layer_one = Dense(channel//ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    
    avg_pool = GlobalAveragePooling2D()(input_feature)    
    avg_pool = Reshape((1,1,channel))(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel//ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel)
    
    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1,1,channel))(max_pool)
    assert max_pool.shape[1:] == (1,1,channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool.shape[1:] == (1,1,channel//ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool.shape[1:] == (1,1,channel)
    
    cbam_feature = Add()([avg_pool,max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)
    
    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)
    
    mul = multiply([input_feature, cbam_feature])
    if type_resnet == 2:
        return mul + input_feature
    return mul

def spatial_attention(input_feature, type_resnet = 0):
    kernel_size = 7
    
    if K.image_data_format() == "channels_first":
        channel = input_feature.shape[1]
        cbam_feature = Permute((2,3,1))(input_feature)
    else:
        channel = input_feature.shape[-1]
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: tf.keras.backend.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool.shape[-1] == 1
    max_pool = Lambda(lambda x: tf.keras.backend.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool.shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat.shape[-1] == 2
    cbam_feature = Conv2D(filters = 1,
                    kernel_size=kernel_size,
                    strides=1,
                    padding='same',
                    activation='sigmoid',
                    kernel_initializer='he_normal',
                    use_bias=False)(concat)	
    assert cbam_feature.shape[-1] == 1
    
    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)
    
    mul = multiply([input_feature, cbam_feature])
    if type_resnet == 2:
        return mul + input_feature
    return mul


#2.Block Define
def basic_block(input, filter_num, stride=1,stage_idx=-1, block_idx=-1, type_resnet = 0):
  '''BasicBlock use stack of two 3x3 convolutions layers

  Args:
    filter_num: the number of filters in the convolution
    stride: the number of strides in the convolution. stride = 1 if you want
            output shape is same as input shape
    stage_idx: index of current stage
    block_idx: index of current block in stage
  '''
  # conv3x3
  conv1=Conv2D(filters=filter_num,
               kernel_size=3,
               strides=stride,
               padding='same',
               kernel_initializer='he_normal',
               name='conv{}_block{}_1_conv'.format(stage_idx, block_idx))(input)
  bn1=BatchNormalization(name='conv{}_block{}_1_bn'.format(stage_idx, block_idx))(conv1)
  relu1=ReLU(name='conv{}_block{}_1_relu'.format(stage_idx, block_idx))(bn1)
  # conv3x3
  conv2=Conv2D(filters=filter_num,
               kernel_size=3,
               strides=1,
               padding='same',
               kernel_initializer='he_normal',
               name='conv{}_block{}_2_conv'.format(stage_idx, block_idx))(relu1)
  bn2=BatchNormalization(name='conv{}_block{}_2_bn'.format(stage_idx, block_idx))(conv2)

  if type_resnet != 0:
    attentionn = cbam_block(cbam_feature = bn2,type_resnet = type_resnet)
    return attentionn
  return bn2


def bottleneck_block(input, filter_num, stride=1, stage_idx=-1, block_idx=-1, type_resnet = 0):
  '''BottleNeckBlock use stack of 3 layers: 1x1, 3x3 and 1x1 convolutions

  Args:
    filter_num: the number of filters in the convolution
    stride: the number of strides in the convolution. stride = 1 if you want
            output shape is same as input shape
    stage_idx: index of current stage
    block_idx: index of current block in stage
  '''
  # conv1x1
  conv1=Conv2D(filters=filter_num,
               kernel_size=1,
               strides=stride,
               padding='valid',
               kernel_initializer='he_normal',
               name='conv{}_block{}_1_conv'.format(stage_idx, block_idx))(input)
  bn1=BatchNormalization(name='conv{}_block{}_1_bn'.format(stage_idx, block_idx))(conv1)
  relu1=ReLU(name='conv{}_block{}_1_relu'.format(stage_idx, block_idx))(bn1)
  # conv3x3
  conv2=Conv2D(filters=filter_num,
               kernel_size=3,
               strides=1,
               padding='same',
               kernel_initializer='he_normal',
               name='conv{}_block{}_2_conv'.format(stage_idx, block_idx))(relu1)
  bn2=BatchNormalization(name='conv{}_block{}_2_bn'.format(stage_idx, block_idx))(conv2)
  relu2=ReLU(name='conv{}_block{}_2_relu'.format(stage_idx, block_idx))(bn2)
  # conv1x1
  conv3=Conv2D(filters=4*filter_num,
               kernel_size=1,
               strides=1,
               padding='valid',
               kernel_initializer='he_normal',
               name='conv{}_block{}_3_conv'.format(stage_idx, block_idx))(relu2)
  bn3=BatchNormalization(name='conv{}_block{}_3_bn'.format(stage_idx, block_idx))(conv3)
  
  if type_resnet != 0:
    attentionn = cbam_block(cbam_feature = bn3,type_resnet = type_resnet)
    return attentionn

  return bn3

def resblock(input, filter_num, stride=1, use_bottleneck=False,stage_idx=-1, block_idx=-1, type_resnet = 0):
  '''A complete `Residual Unit` of ResNet

  Args:
    filter_num: the number of filters in the convolution
    stride: the number of strides in the convolution. stride = 1 if you want
            output shape is same as input shape
    use_bottleneck: type of block: basic or bottleneck
    stage_idx: index of current stage
    block_idx: index of current block in stage
  '''
  if use_bottleneck:
    residual = bottleneck_block(input, filter_num, stride,stage_idx, block_idx, type_resnet = type_resnet)
    expansion=4
  else:
    residual = basic_block(input, filter_num, stride,stage_idx, block_idx, type_resnet = type_resnet)
    expansion=1
  shortcut=input
  # use projection short cut when dimensions increase
  if stride>1 or input.shape[3]!=residual.shape[3]:
    shortcut=Conv2D(expansion*filter_num,
                    kernel_size=1,
                    strides=stride,
                    padding='valid',
                    kernel_initializer='he_normal',
                    name='conv{}_block{}_projection-shortcut_conv'.format(stage_idx, block_idx))(input)
    shortcut=BatchNormalization(name='conv{}_block{}_projection-shortcut_bn'.format(stage_idx, block_idx))(shortcut)

  output=Add(name='conv{}_block{}_add'.format(stage_idx, block_idx))([residual, shortcut])
  return ReLU(name='conv{}_block{}_relu'.format(stage_idx, block_idx))(output)


#3.Stage Define
def stage(input, filter_num, num_block, use_downsample=True, use_bottleneck=False,stage_idx=-1, type_resnet = 0):
  ''' -- Stacking Residual Units on the same stage

  Args:
    filter_num: the number of filters in the convolution used during stage
    num_block: number of `Residual Unit` in a stage
    use_downsample: Down-sampling is performed by conv3_1, conv4_1, and conv5_1 with a stride of 2
    use_bottleneck: type of block: basic or bottleneck
    stage_idx: index of current stage
  '''
  net = resblock(input = input, filter_num = filter_num, stride = 2 if use_downsample else 1, use_bottleneck = use_bottleneck, stage_idx = stage_idx, block_idx = 1, type_resnet = type_resnet)
    
  for i in range(1, num_block):
    net = resblock(input = net, filter_num = filter_num,stride = 1,use_bottleneck = use_bottleneck,stage_idx = stage_idx, block_idx = i+1, type_resnet = type_resnet)
  return net


# 4.Model Define
def build(input_shape,num_classes,layers,use_bottleneck=False,type_resnet = 0):
  '''A complete `stage` of ResNet
  '''
  input=Input(input_shape,name='input')
  # conv1
  net=Conv2D(filters=64,
             kernel_size=7,
             strides=2,
             padding='same',
             kernel_initializer='he_normal',
             name='conv1_conv')(input)
  net=BatchNormalization(name='conv1_bn')(net)
  net=ReLU(name='conv1_relu')(net)
  net=MaxPooling2D(pool_size=3,
                   strides=2,
                   padding='same',
                   name='conv1_max_pool')(net)

  # conv2_x, conv3_x, conv4_x, conv5_x
  filters=[64,128,256,512]
  for i in range(len(filters)):
    net=stage(input = net,
              filter_num = filters[i],
              num_block = layers[i],
              use_downsample = i!=0,
              use_bottleneck = use_bottleneck,
              stage_idx = i+2,
              type_resnet = type_resnet)

  net=GlobalAveragePooling2D(name='avg_pool')(net)
  output=Dense(num_classes,activation='softmax',name='predictions')(net)
  model=Model(input,output)

  return model

def resnet18(input_shape=(224,224,3),num_classes=1000, type_resnet = 0):
  return build(input_shape,num_classes,[2,2,2,2],use_bottleneck=False,type_resnet=type_resnet)

def resnet34(input_shape=(224,224,3),num_classes=1000, type_resnet = 0):
  return build(input_shape,num_classes,[3,4,6,3],use_bottleneck=False,type_resnet=type_resnet)

def resnet50(input_shape=(224,224,3),num_classes=1000, type_resnet = 0):
  return build(input_shape,num_classes,[3,4,6,3],use_bottleneck=True,type_resnet=type_resnet)

def resnet101(input_shape=(224,224,3),num_classes=1000, type_resnet = 0):
  return build(input_shape,num_classes,[3,4,23,3],use_bottleneck=True,type_resnet=type_resnet)

def resnet152(input_shape=(224,224,3),num_classes=1000, type_resnet = 0):
  return build(input_shape,num_classes,[3,8,36,3],use_bottleneck=True,type_resnet=type_resnet)





######################################################  exploring data analysis   #####################################################
# 1.read file exist hdf5 rafdb
class RafdbBasic:
    def __init__(self, root_dir, db_file = "rafdb_basic.hdf5", db_name = "rafdb_basic"):
        self.label_mapping_dict  = {1: "Surprise", 2: "Fear", 3: "Disgust", 4: "Happiness", 5: "Sadness", 6: "Anger", 7: "Neutral"}
        self.label_mapping = np.array(["", "Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"])
        
        self.gender_mapping_dict = {0: "male", 1: "female", 2: "unsure"}
        self.race_mapping_dict   = {0: "Caucasian", 1: "African-American", 2: "Asian"}
        self.age_mapping_dict    = {0: "0-3", 1: "4-19", 2: "20-39", 3: "40-69", 4: "70+"}
        
        self.root_dir= root_dir
        
        self.db_data = pd.read_hdf(db_file, db_name)
        self.db_data = self.db_data.set_index(keys=["id"], drop=False)
        pass # __init__

    def view_image(self, image_id = "train_00010", save_path = None):
        """
        View Image on RAF-DB
        """
        from matplotlib.patches import Rectangle

        org_image_path     = os.path.join(self.root_dir, self.db_data.loc[image_id]["image_original"])
        aligned_image_path = os.path.join(self.root_dir, self.db_data.loc[image_id]["image_aligned"])
        org_image          = cv2.imread(org_image_path)
        aligned_image      = cv2.imread(aligned_image_path)
        image_info         = self.db_data.loc[image_id]

        fig = plt.figure(figsize=(12,10))
        
        ax  = fig.add_subplot(121)

        plt.axis("off")
        ax.imshow(org_image[:,:,::-1])
        ax.plot(image_info["eye1"][0], image_info["eye1"][1], 'ro--', markersize=8)
        ax.plot(image_info["eye2"][0], image_info["eye2"][1], 'ro--', markersize=8)
        ax.plot(image_info["nose"][0], image_info["nose"][1], 'ro--', markersize=8)
        ax.plot(image_info["mouth1"][0], image_info["mouth1"][1], 'ro--', markersize=8)
        ax.plot(image_info["mouth2"][0], image_info["mouth2"][1], 'ro--', markersize=8)
        ax.add_patch(Rectangle((image_info["bbox"][0], image_info["bbox"][1]), 
                               image_info["bbox"][2] - image_info["bbox"][0], 
                               image_info["bbox"][3] - image_info["bbox"][1], fill=False, ec="red", lw=2.0))

        for point in image_info["landmarks"]:
            ax.plot(point[0], point[1], 'bo--', markersize=5)
            
            pass # for

        s_text = "%s\n%s\n%s\n%s"%(self.label_mapping_dict[image_info["emotion"]], 
                                self.gender_mapping_dict[image_info["gender"]],
                                self.age_mapping_dict[image_info["age"]], 
                                self.race_mapping_dict[image_info["race"]])
        
        plt.text(image_info["bbox"][2] + 5, image_info["bbox"][3], s_text, color = "r", fontsize=18)
        
        ax  = fig.add_subplot(122)
        plt.axis("off")
        # ax.plot(25, 33, 'ro--', markersize=8)
        # ax.plot(75, 33, 'ro--', markersize=8)
        # ax.plot(50, 73, 'ro--', markersize=8)
        ax.imshow(aligned_image[:,:,::-1])

        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()
        plt.close()
        
        return fig
        pass # view_image

    pass # RafdbBasic


# 2.Data augmentation
def train_aug(image_size, p=1.0):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.PadIfNeeded(min_height=image_size[0], min_width=image_size[1], p=0.5),
        A.Rotate(limit=(-90, 90), p=0.8) 
],p = 0.95)

def transform_image(image):
    image = image.numpy()  # Convert to numpy array
    transformed = train_aug(image.shape[:2])(image=image)['image']
    #return transformed.astype(np.float32)
    return transformed

def train_aug_horizontal(image_size, p=1.0):
    return A.Compose([
        A.HorizontalFlip(p=1),
])

def train_aug_verticalFlip(image_size, p=1.0):
    return A.Compose([
        A.VerticalFlip(p=1),
])

def train_aug_randomBrightnessContrast(image_size, p=1.0):
    return A.Compose([
        A.RandomBrightnessContrast(p=1)
])

def train_aug_left45(image_size, p=1.0):
    return A.Compose([
        A.Rotate(limit=(45, 45), p=1) 
])

def train_aug_right45(image_size, p=1.0):
    return A.Compose([
        A.Rotate(limit=(-45, -45), p=1) 
])
