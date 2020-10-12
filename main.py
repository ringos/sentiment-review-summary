from tensorboardX import SummaryWriter
import torch.utils.data as tdata
import pickle
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.tokenize import sent_tokenize,TreebankWordTokenizer
import random
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import argparse
from importlib import import_module

import processing_data
import models
import functions

#*****************************************************************
#  set random seed
seed=42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
def _init_fn(worker_id):
    ''' for dataloader workers init, freeze dataloader's randomness '''
    np.random.seed(seed + worker_id)
#***************************************************************

tokenizer=TreebankWordTokenizer()

def adjust_lr(optimizer, epoch, init_lr, decay_rate=0.3, decay_interval=20):
    lr = init_lr * (decay_rate ** (epoch // decay_interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='config file')
    args = parser.parse_args()
    param=import_module('param.'+args.config)
    if param.MODE not in {'dev','test','both'}:
        raise ValueError('Mode \'{}\' does not exist'.format(param.MODE))
    if param.DATASET_NAME==param.TESTSET_NAME:
        train_loader,dev_loader,test_loader=processing_data.load_dataset(param,tokenizer,param.DATASET_NAME, param.DATA_MODE, decode_folder=param.ROOT_DATA_PATH+'sentiment/pointer_generator/decode_folder/')
    else:
        train_loader,dev_loader=processing_data.load_dataset(param,tokenizer,param.DATASET_NAME, param.DATA_MODE)[:2]
        test_loader=processing_data.load_dataset(param,tokenizer,param.TESTSET_NAME, param.DATA_MODE)[2]
    
    # if param.DATA_MODE == 'predicted': 
    #     test_loader=processing_data.load_dataset(param,tokenizer,param.DATASET_NAME, 'predicted', decode_folder=param.ROOT_DATA_PATH+'sentiment/pointer_generator/decode_folder/')[2]

    criterion=nn.CrossEntropyLoss().to('cuda')


    model=models.BiLSTM_centric_model(param.INPUT_SIZE,
                                    param.HIDDEN_SIZE,
                                    param.OUTPUT_CLASSES,
                                    num_layers=param.NUM_LAYERS,
                                    num_heads=param.NUM_HEADS,
                                    dropout_rate=param.DROPOUT_RATE,
                                    use_residual=param.USE_RESIDUAL,
                                    use_concate_raw=param.USE_CONCATE_RAW,
                                    use_concate_sum=param.USE_CONCATE_SUM,
                                    use_layer_norm=param.USE_LAYER_NORM,
                                    use_divide_dk=param.USE_DIVIDE_DK).to('cuda')
    if param.OPTIMIZER=='sgd':
        optimizer=optim.SGD(model.parameters(),lr=param.LR)
    elif param.OPTIMIZER=='adam':
        optimizer=optim.Adam(model.parameters(),lr=param.LR)
    elif param.OPTIMIZER=='sgd_with_momentum':
        optimizer=optim.SGD(model.parameters(),lr=param.LR,momentum=0.9)
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[(i+1)*15 for i in range(10)], gamma=0.333)

    print("-----Start training-----\n")
    start_time=time.time()
    record_path=param.RECORD_PATH+args.config+'/'  #create a new folder for the record
    try:
        os.mkdir(record_path)
        print('\nCreating record folder: {}'.format(args.config))
    except FileExistsError as e:
        print('\nWarning: the record folder exists. Maybe there is something wrong with: {0}.\n'.format(args.config))
    writer=SummaryWriter(record_path)

    # num_total_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('\n-----model total parameters: {}'.format(num_total_parameters))
    # writer.add_text('total parameters','total_parameters: {}'.format(num_total_parameters))

    epoch_interval=5
    batch_interval=100
    num_batch=len(train_loader)

    current_acc_dev=0.0
    current_acc_test=0.0
    highest_acc_test=0.0

    for epoch in range(param.EPOCH_NUM):
#         '''
#         adaptive Learning Rate
#         '''
#         if epoch>2:
#             optimizer=optim.Adam(model.parameters(),lr=LR/(5**(epoch-2)))
#         else:
#             optimizer=optim.Adam(model.parameters(),lr=LR)
        adjust_lr(optimizer,epoch,param.LR,decay_rate=param.DECAY_RATE,decay_interval=param.DECAY_INTERVAL)

        s_time=time.time()
        writer.add_text('time','Starting time:{0}'.format(time.asctime()),epoch)
    
        running_loss=0.0
        total_loss=0.0

        total_seq_len=0

        for i,(batch_raw,batch_label,batch_sum) in enumerate(train_loader):
    
            batch_raw_ids=batch_raw[0].to('cuda')
            batch_raw_len=batch_raw[1].to('cuda')

            total_seq_len+=batch_raw_len.max().item()

            batch_label=batch_label.to('cuda').to(dtype=torch.long)
            if param.NOT_USE_SIMPLE:
                batch_sum_ids=batch_sum[0].to('cuda')
                batch_sum_len=batch_sum[1].to('cuda')
                
            model.train()
    
            model.zero_grad()
            optimizer.zero_grad()

            if param.NOT_USE_SIMPLE:
                prediction=model(batch_raw_ids,batch_sum_ids,batch_raw_len,batch_sum_len,use_sum_avg_pooling=param.USE_SUM_AVG_POOLING)
            else:
                prediction=model(batch_raw_ids,batch_raw_len)
    
            loss=criterion(prediction,batch_label)
    
            running_loss+=loss.item()
            total_loss+=loss.item()

            writer.add_scalar('batch_loss',loss.item()/param.BATCH_SIZE,i+1+epoch*num_batch)

            loss.backward()
            optimizer.step()
            # scheduler.step()
            # print('current LR={}'.format(scheduler.get_lr()))

            if i%batch_interval==batch_interval-1:
                print('{0}th epoch {1}th batch: loss={2}'.format(epoch+1,i+1,running_loss/param.BATCH_SIZE/batch_interval),
                        '{0} batches time spent: {1} seconds'.format(batch_interval,time.time()-s_time),
                        '*****{0} batches average max_len={1}'.format(batch_interval,total_seq_len/batch_interval))
                # print('optimizer_parameters: ',optimizer.state_dict()['lr'])
                total_seq_len=0
                s_time=time.time()

                if param.MODE is 'test':
                    tst=functions.test(model,test_loader,param.NOT_USE_SIMPLE,param.USE_SUM_AVG_POOLING)
                elif param.MODE is 'dev':
                    tst=functions.test(model,dev_loader,param.NOT_USE_SIMPLE,param.USE_SUM_AVG_POOLING)
                elif param.MODE is 'both':
                    tst_dev=functions.test(model,dev_loader,param.NOT_USE_SIMPLE,param.USE_SUM_AVG_POOLING)
                    tst_test=functions.test(model,test_loader,param.NOT_USE_SIMPLE,param.USE_SUM_AVG_POOLING)
                if param.MODE in {'dev','test'}:
                    print('{0}th batch testing result on {1}:\n'.format(i+1,param.DATASET_NAME),tst)
                    for pp,_ in enumerate(tst.items()):
                        writer.add_scalar(_[0],_[1],i+1+epoch*num_batch)
                else:
                    print('{0}th batch testing result on dev:\n'.format(i+1),tst_dev,'\n')
                    print('{0}th batch testing result on test:\n'.format(i+1),tst_test,'\n')
                    if highest_acc_test<tst_test['accuracy']:
                        highest_acc_test=tst_test['accuracy']
                    if tst_dev['accuracy']>current_acc_dev:
                        current_acc_test=tst_test['accuracy']
                        current_acc_dev=tst_dev['accuracy']
                    elif tst_dev['accuracy']==current_acc_dev and current_acc_test<tst_test['accuracy']:
                        current_acc_test=tst_test['accuracy']
                        current_acc_dev=tst_dev['accuracy']
                    print('current test_acc={0} based on dev_acc={1} \ncurrent highest test_acc: {2}\n\n*********************************************\n'.format(current_acc_test,current_acc_dev,highest_acc_test))

                    for pp,_ in enumerate(tst_dev.items()):
                        writer.add_scalar(_[0]+'_dev',_[1],i+1+epoch*num_batch)
                    for pp,_ in enumerate(tst_test.items()):
                        writer.add_scalar(_[0]+'_test',_[1],i+1+epoch*num_batch)
                        running_loss=0.0

        if epoch%epoch_interval==epoch_interval-1:
            torch.save({'epoch':epoch,
                            'model_state_dict':model.state_dict(),
                            'optimizer_state_dict':optimizer.state_dict()
                            },record_path+args.config+'_{}th_epoch'.format(epoch+1))
        # record_file.writelines('\n{0}th epoch overall loss={1} \n'.format(epoch+1,total_loss/i/param.BATCH_SIZE))
        print("{0}th epoch: average loss={1} \n".format(epoch+1,total_loss/num_batch/param.BATCH_SIZE))
        writer.add_scalar('epoch_average_loss',total_loss/i/param.BATCH_SIZE,epoch+1)

        test_time=time.time()
        model.eval()
        print('----------tesing on {} set----------\n'.format(param.TESTSET_NAME))
        tst_train=functions.test(model,train_loader,param.NOT_USE_SIMPLE,param.USE_SUM_AVG_POOLING)
        if param.MODE is 'test':
            tst=functions.test(model,test_loader,param.NOT_USE_SIMPLE,param.USE_SUM_AVG_POOLING)
        elif param.MODE is 'dev':
            tst=functions.test(model,dev_loader,param.NOT_USE_SIMPLE,param.USE_SUM_AVG_POOLING)
        elif param.MODE is 'both':
            tst_dev=functions.test(model,dev_loader,param.NOT_USE_SIMPLE,param.USE_SUM_AVG_POOLING)
            tst_test=functions.test(model,test_loader,param.NOT_USE_SIMPLE,param.USE_SUM_AVG_POOLING)
        print('\n{0}th epoch testing result on train_set:\n'.format(epoch+1),tst_train,'\n')
        if param.MODE in {'dev','test'}:
            print('{0}th epoch testing result on {1}:\n'.format(epoch+1,param.DATASET_NAME),tst,'\n')
            for pp,_ in enumerate(tst.items()):
                writer.add_scalar(_[0],_[1],epoch+1)
        else:
            print('{0}th epoch testing result on dev:\n'.format(epoch+1),tst_dev,'\n')
            print('{0}th epoch testing result on test:\n'.format(epoch+1),tst_test,'\n')
            if highest_acc_test<tst_test['accuracy']:
                highest_acc_test=tst_test['accuracy']
            if tst_dev['accuracy']>current_acc_dev:
                current_acc_test=tst_test['accuracy']
                current_acc_dev=tst_dev['accuracy']
            elif tst_dev['accuracy']==current_acc_dev and current_acc_test<tst_test['accuracy']:
                current_acc_test=tst_test['accuracy']
                current_acc_dev=tst_dev['accuracy']
            print('current test_acc={0} based on dev_acc={1} \ncurrent highest test_acc: {2}\n\n*********************************************\n'.format(current_acc_test,current_acc_dev,highest_acc_test))
            
            for pp,_ in enumerate(tst_train.items()):
                writer.add_scalar(_[0]+'_train_epoch',_[1],epoch+1)
            for pp,_ in enumerate(tst_dev.items()):
                writer.add_scalar(_[0]+'_dev_epoch',_[1],epoch+1)
            for pp,_ in enumerate(tst_test.items()):
                writer.add_scalar(_[0]+'_test_epoch',_[1],epoch+1)
            
            print('\ntest spent time: {} seconds\n'.format(time.time()-test_time))