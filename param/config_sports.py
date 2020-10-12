# -*- coding: utf-8 -*-


ROOT_DATA_PATH='the_path_of_your_data_folder'
FILE_PATH=ROOT_DATA_PATH+'sentiment/amazon_review/'
RECORD_PATH=ROOT_DATA_PATH+'sentiment/record/'
GLOVE_PATH=ROOT_DATA_PATH+'glove_model/glove.840B.300d.pkl'


DATASET_ALL={'toy','sports','movie'}
TOY_FILE='reviews_Toys_and_Games_5.json.gz'
MOVIE_FILE='reviews_Movies_and_TV_5.json.gz'
SPORTS_FILE='reviews_Sports_and_Outdoors_5.json.gz'


INPUT_SIZE=300
HIDDEN_SIZE=256
OUTPUT_CLASSES=5

NUM_LAYERS=2
NUM_HEADS=1

DROPOUT_RATE=0.2
USE_LAYER_NORM=True
USE_RESIDUAL=True
USE_CONCATE_RAW=False
USE_CONCATE_SUM=False
USE_SUM_AVG_POOLING=True
USE_DIVIDE_DK=True

EPOCH_NUM=10
BATCH_SIZE=128

#choose from 'sgd', 'sgd_with_momentum' or 'adam'
OPTIMIZER='adam'

DATA_MODE = 'golden'

##########################################################
DATASET_NAME='sports'   #choose from 'toy', 'sports' and 'movie'
TESTSET_NAME='sports'
USE_SUMMARY=True   # Only required when NOT_USE_SIMPLE==False. True - use raw and summary, False - only use raw
NOT_USE_SIMPLE=True  # whether to use BiLSTM-review-centric or BiLSTM-vanilla  True: not use vanilla; false: use vanilla
ONLY_USE_SUMMARY=False
MODE='both'  # choose from 'dev' or 'test'


##########################################################
LR=3e-4
DECAY_RATE=1
DECAY_INTERVAL=200

