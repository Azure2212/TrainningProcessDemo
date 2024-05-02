from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D,BatchNormalization, AveragePooling2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy, CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adamax
from argparse import ArgumentParser
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, BackupAndRestore, LambdaCallback
import pickle
import matplotlib.pyplot as plt
import os, sys
from datetime import datetime
import numpy as np
import pytz
import json
import keras
import tensorflow as tf
import wandb
from wandb.keras import WandbMetricsLogger

import sys
sys.path.append("/kaggle/working/sgu24project/resnet_model/")
import resnet_model

current_time = datetime.now()

# Convert to Vietnam time zone
vietnam_timezone = pytz.timezone('Asia/Ho_Chi_Minh')
current_time_vietnam = current_time.astimezone(vietnam_timezone)

# Format the datetime object to exclude milliseconds
current_time = str(current_time_vietnam.strftime('%Y-%m-%d %H:%M:%S'))

current_time = current_time.replace(" ","_")
current_time = current_time.replace("-","_")
current_time = current_time.replace(":","_")
current_time = current_time.replace("+","_")

print(current_time)
path_current = os.path.abspath(globals().get("__file__","."))
script_dir  = os.path.dirname(path_current)
print(script_dir)
root_path = os.path.abspath(f"{script_dir}/../../../")
experiments_dir = os.path.abspath(f"/kaggle/working/exps/Resnet/experiment_{current_time}")
data_path = "/kaggle/input/rafdb-basic-after-clustering/rafdb_basic_after_clustering"
print(experiments_dir)
print(data_path)

import argparse
#sys.argv = ["/kaggle/working/sgu24project/test.ipynb", "--model", "tuandeptrai","--model2","happy"] parsing args through .ipynb
import json
# Read the JSON file
with open(r'/kaggle/working/sgu24project/resnet_model/resnetParams.json') as f:
    data = json.load(f)
all_inital_argument = data

parser = argparse.ArgumentParser()
parser.add_argument('--model', default= all_inital_argument['model']['choose'], type=str, help='Type of model')
parser.add_argument('--train-folder', default= all_inital_argument['train-folder'], type=str, help='Where training data is located')
parser.add_argument('--valid-folder', default= all_inital_argument['valid-folder'], type=str, help='Where validation data is located')
parser.add_argument('--num-classes', default= all_inital_argument['num-classes'], type=int, help='Number of classes')
parser.add_argument("--batch-size", default= all_inital_argument['batch-size'], type=int)
parser.add_argument('--image-size', default= all_inital_argument['image-size'], type=int, help='Size of input image') # initial 224
parser.add_argument('--optimizer', default= all_inital_argument['optimizer']['choose'], type=str, help='Types of optimizers')
parser.add_argument('--lr', default=all_inital_argument['lr'], type = float, help='Learning rate')
parser.add_argument('--epochs', default= all_inital_argument['epochs'], type=int, help = 'Number of epochs')
parser.add_argument('--image-channels', default= all_inital_argument['image-chennels'], type=int, help='Number channel of input image')
parser.add_argument('--class-mode', default= all_inital_argument['class-mode']['choose'], type=str, help='Class mode to compile')
parser.add_argument('--model-path', default= all_inital_argument['model-path'], type=str, help='Path to save trained model')
    
parser.add_argument('--exp-dir', default = experiments_dir, type=str, help='folder contain experiemts')
parser.add_argument('--author-name', default= all_inital_argument['author-name'], type=str, help='name of an author')
parser.add_argument('--use-wandb', default= all_inital_argument['use-wandb'], type=int, help='Use wandb')
parser.add_argument('--debug', default= all_inital_argument['debug']['choose'], type=int, help='debug')
parser.add_argument('--transform-type', default = all_inital_argument['transform-type'], type=str, help='folder contain experiemts')
parser.add_argument('--wandb-api-key', default = all_inital_argument['wandb-api-key'], type=str, help='wantdb api key')
parser.add_argument('--wandb-project-name', default = all_inital_argument['wandb-project-name'], type=str, help='name project to store data in wantdb')
parser.add_argument('--d-steps', default= all_inital_argument['d-steps'], type=int, help='step per epochs')
parser.add_argument('--type-resnet', default= all_inital_argument['type-resnet']['choose'], type=int, help='0: Resnet, 1: CBAM_Resnet, 2: residual_resnet')
parser.add_argument('--early-stopping', default= all_inital_argument['early-stopping'], type=int, help='early stopping if n e-pochs not better!')
parser.add_argument('--lr-decrease', default= all_inital_argument['lr-decrease'], type=int, help='value learning rate decrease after 5 epochs not better')
args, unknown = parser.parse_known_args()
configs = vars(args)



training_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2)
val_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = training_datagen.flow_from_directory(args.train_folder, target_size=(args.image_size, args.image_size), batch_size= args.batch_size, class_mode = args.class_mode, shuffle = True)
val_generator = val_datagen.flow_from_directory(args.valid_folder, target_size=(224, 224), batch_size= args.batch_size, class_mode = args.class_mode, shuffle = True)


# Create model
if args.model == 'resnet18':
    model = resnet_model.resnet18(num_classes = args.num_classes, type_resnet = args.type_resnet)
elif args.model == 'resnet34':
    model = resnet_model.resnet34(num_classes = args.num_classes, type_resnet = args.type_resnet)
elif args.model == 'resnet50':
    model = resnet_model.resnet50(num_classes = args.num_classes, type_resnet = args.type_resnet)
elif args.model == 'resnet101':
    model = resnet_model.resnet101(num_classes = args.num_classes, type_resnet = args.type_resnet)
elif args.model == 'resnet152':
    model = resnet_model.resnet152(num_classes = args.num_classes, type_resnet = args.type_resnet)
else:
    print('Wrong resnet name, please choose one of these model: resnet18, resnet34, resnet50, resnet101, resnet152')
    raise Exception("model resnet incorrect")
# save all arguments to json file
print(args.exp_dir)
os.makedirs(args.exp_dir, exist_ok=True)

args_dict = vars(args)

# Write dictionary to file
file_path = os.path.join(experiments_dir, "arguments.json")
print(file_path)
with open(file_path, 'w') as file:
    json.dump(args_dict, file, indent=4)

model.build(input_shape=(None, args.image_size, args.image_size, args.image_channels))
#model.summary()


if (args.optimizer == 'adam'):
    optimizer = Adam(learning_rate=args.lr)
elif (args.optimizer == 'sgd'):
    optimizer = SGD(learning_rate=args.lr)
elif (args.optimizer == 'rmsprop'):
    optimizer = RMSprop(learning_rate=args.lr)
elif (args.optimizer == 'adadelta'):
    optimizer = Adadelta(learning_rate=args.lr)
elif (args.optimizer == 'adamax'):
    optimizer = Adamax(learning_rate=args.lr)
else:
    raise 'Invalid optimizer. Valid option: adam, sgd, rmsprop, adadelta, adamax'

model.compile(optimizer=optimizer, 
            loss=SparseCategoricalCrossentropy(),
            metrics=['accuracy'])


from keras.callbacks import Callback,ReduceLROnPlateau
# callbacks
callbacks = []

# wandb
if args.use_wandb == 1:
    cb_wandb = WandbMetricsLogger(log_freq=1)
    callbacks.append(cb_wandb)

# best_model
save_bestmodel_path = f'{args.exp_dir}/{args.model_path}'
cb_best_model = ModelCheckpoint(save_bestmodel_path,
                             save_weights_only=False,
                             monitor='val_loss',
                             verbose=1,
                             mode='min',
                             save_best_only=True)
callbacks.append(cb_best_model)

# backup
backup_dir = f'{args.exp_dir}'
cb_backup = BackupAndRestore(backup_dir)
#callbacks.append(cb_backup)

# logger
cb_log = CSVLogger(f'{args.exp_dir}/log.csv')
callbacks.append(cb_log)
#early stopping
callbacks.append(keras.callbacks.EarlyStopping(monitor='val_loss',patience=args.early_stopping))       
          
# learning rate reduce after 5 epchs without loss decrease
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
callbacks.append(lr_scheduler)

steps_per_epoch = None
epochs = args.epochs
if args.debug == 1:
    steps_per_epoch = args.d_steps


model.fit(
    train_generator,
    epochs=epochs,
    verbose=1,
    steps_per_epoch = steps_per_epoch,
    validation_data=val_generator,
    callbacks=callbacks,
    validation_steps= steps_per_epoch
    )

# 🐝 Close your wandb run 
if args.use_wandb == 1:
    wandb.finish()