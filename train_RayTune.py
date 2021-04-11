#!/usr/bin/env python3
from Plotter_CellSpike import Plotter_CellSpike
from Deep_CellSpike import Deep_CellSpike
from Util_IOfunc import write_yaml

import tensorflow as tf
Lpd=tf.config.list_physical_devices('GPU')
print('Lpd, devCnt=',len(Lpd), Lpd)

import argparse

# get_parser copied from train_CellSpike.py. This will allow us to specify the same arguments as when we run train_CellSpike.py

def get_parser():
    '''This is the same function (same name) from train_CellSpike.py. No modifications needed for now.'''

    parser = argparse.ArgumentParser()
    
    parser.add_argument("-n","--nodes",help="running on CPU or GPU nodes",
        default='CPU')
    
    parser.add_argument("--numCPU",help="number of CPUs for each trial",
        default=30)
    parser.add_argument("--numGPU",help="numuber of GPUs for each trial",
        default=1)
        
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2],
        help="increase output verbosity", default=1, dest='verb')

    parser.add_argument('--design', dest='modelDesign', default='Raytune_design',
        help=" model design of the network")

    parser.add_argument('--rayResult', dest='rayResult', default='./ray_results',
        help="the output directory of raytune")
    parser.add_argument('--numHparams', dest='numHparams', default='5',
        help="the number of Raytune Samples")

    parser.add_argument('--venue', dest='formatVenue', choices=['prod','poster'], default='prod',help=" output quality/arangement")

    parser.add_argument("--seedWeights",default=None,
        help="seed weights only, after model is created")

    parser.add_argument("-d","--dataPath",help="path to input",
        default='data')
    parser.add_argument("--probeType",  help="probe partition or PCA",default='pca99')

    parser.add_argument("-o","--outPath",
        default='out',help="output path for plots and tables")

    parser.add_argument( "-X","--noXterm", dest='noXterm',
        action='store_true', default=False,
        help="disable X-term for batch mode")

    parser.add_argument("-t", "--trainTime", type=int, default=20,
        help="training time (seconds)")

    parser.add_argument( "--maxEpochTime", type=int, default=None,
        help="maximal allowed training/epoch time (seconds)")

    parser.add_argument("-b", "--batch_size", type=int, default=None,
        help="training batch_size")
    parser.add_argument( "--steps", type=int, default=None,
        help="overwrite natural steps per training epoch, default:None, set by generator")
    parser.add_argument( "--numFeature", type=int, default=None, nargs='+',
        help="input features, overwrite hpar setting, can be a list")
    parser.add_argument("--useDataFrac", type=float, default=1.,
        help="InpGen will use fraction of whole dataset, default is use all data")

    parser.add_argument("--dropFrac", type=float, default=None,
        help="drop fraction at all FC layers, default=None, set by hpar")
    parser.add_argument( "-s","--earlyStop", type=int,
        dest='earlyStopPatience', default=10,
        help="early stop:  epochs w/o improvement (aka patience), 0=off")
    parser.add_argument( "--checkPt", dest='checkPtOn',
        action='store_true',default=True,help="enable check points for weights")

    parser.add_argument( "--reduceLR", dest='reduceLRPatience', type=int,
        default=5,help="reduce learning at plateau, patience")

    parser.add_argument("-j","--jobId", default=None,
        help="optional, aux info to be stored w/ summary")

    args = parser.parse_args()
    args.train_loss_EOE=True #True # 2nd loss computation at the end of each epoch
    args.shuffle_data=True # use False only for debugging

    args.prjName='cellSpike'
    args.dataPath+='/'
    args.outPath+='/'
    if args.numFeature!=None:
        # see Readme.numFeature for the expanation
        if len(args.numFeature)==1: # Warn: selecting 1 feature which is not soma is not possible with this logic
            args.numFeature=args.numFeature[0]
        else: #assure uniqnenss of elements
            assert len(args.numFeature)==len(set(args.numFeature))

    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))

    return args

# Imports the TuneReportCheckpointCallback class, which will handle checkpointing and reporting for us.

import ray
from ray import tune, init
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.integration.keras import TuneReportCheckpointCallback

import numpy as np

import os, time
import warnings
import socket  # for hostname
import copy
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings

startT0 = time.time()
from Util_IOfunc import write_yaml, read_yaml
from Util_CellSpike import MyLearningTracker, MyEpochEndLoss

from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint , ReduceLROnPlateau
from tensorflow.python.keras.layers import Dense, Dropout,  Input, Conv1D,MaxPool1D,Flatten, Lambda, BatchNormalization
from  tensorflow.python.keras.layers.advanced_activations import LeakyReLU

# pick one of this data streamers:
from InpGenDisc_CellSpike import  CellSpike_input_generator
#from InpGenRam_CellSpike import  CellSpike_input_generator

import tensorflow as tf

import numpy as np
import h5py

from numpy import linalg as LA  # for eigen vals
from pprint import pprint
#--------

class RayTune_CellSpike(Deep_CellSpike):



    @classmethod #........................
    def trainer(cls, config, args):
        print('Cnst:train')
        obj=cls(**vars(args))
        obj.read_metaInp(obj.dataPath+obj.metaF)
        obj.train_hirD={'acc': [],'loss': [],'lr': [],'val_acc': [],'val_loss': []}

        obj.hparams = config


        # overwrite some hpar if provided from command line
        if obj.dropFrac!=None:
            obj.hparams['dropFrac']=obj.dropFrac
        if obj.batch_size!=None:
            obj.hparams['batch_size']=obj.batch_size
        if obj.steps!=None:
            obj.hparams['steps']=obj.steps

        if obj.numFeature!=None:
            obj.hparams['numFeature']=obj.numFeature

        # sanity checks
        assert obj.hparams['dropFrac']>=0

        # fix None-strings
        for x in obj.hparams:
            if obj.hparams[x]=='None': obj.hparams[x]=None
        obj.sumRec={}
        return obj


    def train_model(self):
        '''
        ---- This method overrides the one from Deep_CellSpike. It is almost exactly identical
        ---- except the callbacks_list now contains a
        ---- TuneReportCheckpointCallback() object The TuneReportCheckpointCallback
        ---- class handles reporting metrics and checkpoints. The metric it reports is validation loss.

        '''
        print('training initiating')
        if self.verb==0: print('train silently ')

        callbacks_list = [TuneReportCheckpointCallback(metrics = ['val_loss'])] #callbacks_list contains a TuneReportCheckpointCallback object that handles checkpointing and reporting the validation loss to Tune.
        print('callbacks_list initiated')
        lrCb=MyLearningTracker()
        callbacks_list.append(lrCb)

        trlsCb=None
        if self.train_loss_EOE:
            print('enable  end-epoch true train-loss as callBack')
            genConf=copy.deepcopy(self.sumRec['inpGen']['train'])
            # we need much less stats for the EOE loss:
            genConf['fakeSteps']=self.sumRec['inpGen']['val']['fakeSteps']
            fileIdxL=genConf['fileIdxL']
            if type(fileIdxL)==type(list()): #data-stream
                nSample=int( len(fileIdxL)/10.)+1  # select only 10% of files
                fileIdxL=np.random.choice(fileIdxL, nSample, replace=False).tolist()
            else: # data-in-RAM
                fileIdxL=max(1, fileIdxL//20)  # select 5% of frames
            genConf['fileIdxL']=fileIdxL
            genConf['name']='trainEOE'
            inpGen=CellSpike_input_generator(genConf,verb=1)
            trlsCb=MyEpochEndLoss(inpGen)
            callbacks_list.append(trlsCb)

        if self.earlyStopPatience>0:
            earlyStop=EarlyStopping(monitor='val_loss', patience=self.earlyStopPatience, verbose=1, min_delta=2.e-4, mode='auto')
            callbacks_list.append(earlyStop)
            print('enabled EarlyStopping, patience=',self.earlyStopPatience)

        if self.checkPtOn:
            outF5w=self.outPath+'/'+self.prjName+'.weights_best.h5'
            chkPer=1
            ckpt=ModelCheckpoint(outF5w, save_best_only=True, save_weights_only=True, verbose=1, period=chkPer,monitor='val_loss')
            callbacks_list.append(ckpt)
            print('enabled ModelCheckpoint, save_freq=',chkPer)

        if self.reduceLRPatience>0:
            redu_lr = ReduceLROnPlateau(monitor='val_loss', factor=self.hparams['reduceLR_factor'], patience=self.reduceLRPatience, min_lr=0.0, verbose=1,min_delta=0.003)
            callbacks_list.append(redu_lr)
            print('enabled ReduceLROnPlateau, patience=',self.reduceLRPatience,',factor=',self.hparams['reduceLR_factor'])

        print('\nTrain_model  trainTime=%.1f(min)'%(self.trainTime/60.),'  modelDesign=', self.modelDesign,'BS=',self.hparams['batch_size'])

        fitVerb=1   # prints live:  [=========>....] - ETA: xx s
        if self.verb==2: fitVerb=1
        if self.verb==0: fitVerb=2  # prints 1-line summary at epoch end
        train_state='model_trained'
        for st in range(2): # state-machine
            startTm = time.time()
            print('jan train state:',st)
            if st==0:
                epochs=2 ; startTm0=startTm ;  totEpochs=epochs
            if st==1:
                print(' used time/sec=%.1f'%fitTime)
                maxEpochs=int(0.95*epochs*self.trainTime/fitTime) -2  # 0.95 is contingency
                if self.maxEpochTime!=None:
                    if fitTime/epochs > self.maxEpochTime :
                        print('too slow training, abort it', fitTime/epochs, self.maxEpochTime)
                        train_state='epoch_to_slow'
                        totEpochs-=1; break
                if maxEpochs<1 :
                    print('not enough time to run more, finish')
                    train_state='time_limit_reached'
                    totEpochs-=1; break
                epochs=maxEpochs
                print('will train for %d epochs over total of %.1f minutes'%(epochs,self.trainTime/60.))
            # common for both steps:
            #            hir=self.model.fit_generator(  #TF-2.0
            hir=self.model.fit(  # TF-2.1
                    self.inpGenD['train'],callbacks=callbacks_list,
                    epochs=epochs, max_queue_size=10,
                    workers=1, use_multiprocessing=False,
                    shuffle=self.shuffle_data, verbose=fitVerb,
                    validation_data=self.inpGenD['val'])
            print('model fit complete')
            fitTime=time.time() - startTm
        # correct the fit time  to account for all process
        fitTime=time.time() - startTm0
        hir=hir.history
        epochs2=len(hir['loss'])
        totEpochs+=epochs2
        earlyStopOccured=epochs2<epochs

        #print('FFF',earlyStopOccured,epochs2, maxEpochs,totEpochs)
        #print('hir keys',hir.keys(),lrCb.hir)
        for obs in hir:
            rec=[ float(x) for x in hir[obs] ]
            self.train_hirD[obs].extend(rec)

        # this is a hack, 'lr' is returned by fit only when --reduceLr is used
        if 'lr' not in  hir:
            self.train_hirD['lr'].extend(lrCb.hir)

        if trlsCb: # add train-loss for Kris
            nn=len(self.train_hirD['loss'])
            self.train_hirD['train_loss']=trlsCb.hir[-nn:]

        self.train_hirD['stepTimeHist']=self.inpGenD['train'].stepTime

        #report performance for the last epoch
        lossT=self.train_hirD['loss'][-1]
        lossV=self.train_hirD['val_loss'][-1]
        lossVbest=min(self.train_hirD['val_loss'])

        hpar=self.hparams
        print('\n End Val Loss=%s:%.3f, best:%.3f'%(hpar['lossName'],lossV,lossVbest),', %d totEpochs, fit=%.1f min'%(totEpochs,fitTime/60.),' hpar:',hpar)
        self.train_sec=fitTime

        # add info to summary
        rec={}
        rec['earlyStopOccured']=int(earlyStopOccured)
        rec['fitTime_min']=fitTime/60.
        rec['totEpochs']=totEpochs
        rec['train_loss']=float(lossT)
        rec['val_loss']=float(lossV)
        rec['steps_per_epoch']=self.inpGenD['train'].__len__()
        rec['state']=train_state
        self.sumRec.update(rec)


args=get_parser()

def training_initialization():
    """
    The parent function count the number of times a model is created
    which then creates a folder for the output of each model
    """
    count = 0
    model_path = args.outPath
    print("current work dir = " + os.getcwd())
    print("All output of Deep_CellSpike model will be saved to: " + str(model_path))

    #if not os.path.exists(model_path):
    #    os.makedirs(model_path)

    def make_folder(count):
        model_n_path = model_path + "/model_" + str(count)
        print("current work dir = " + os.getcwd())
        print("Model_" + str(count) + " will be save to: " + str(model_n_path))

        if not os.path.exists(model_n_path):
            os.makedirs(model_n_path)

        args.outPath = model_n_path



    def training_function(config, checkpoint_dir = None):
        
        
        nonlocal count
        count += 1
        make_folder(count)
        
        config['myID'] = 'id1' + ('%.11f'%np.random.uniform())[2:]
        
        config['conv_kernel'] = int(config['conv_kernel'])
        config['conv_repeat'] = int(config['conv_repeat'])
        config['pool_len'] = int(config['pool_len'])
        
        conv_filter = []
        if not config["conv_filter"]:
            cf_num_layers = int(config["cf_num_layers"])
            pwr = 0  # Maybe start with a higher pwr here
            for i in range(1, cf_num_layers + 1):
                additional_pwr = int(config[f"filter_{i}"])
                pwr += additional_pwr
                filter_size = 2**pwr
                conv_filter.append(min(256, filter_size))
            config["conv_filter"] = conv_filter
        
        
        if not config["fc_dims"]:
            fc_dims = []
            fc_num_layers = int(config["fc_num_layers"])
            pwr2 = 0
            for i in range(1, fc_num_layers + 1):
                additional_pwr2 = config[f"fc_{i}"]
                pwr2 += additional_pwr2
                fc_size = 2**pwr2
                fc_dims.insert(0, min(512, fc_size))
            config["fc_dims"] = fc_dims
        
        if not config["optimizer"]:
            optimizer = [config["optName"], 0.001, 1.1e-7]
            config["optimizer"] = optimizer
        
        if not config["batch_size"]:
            batch_size = 1<<int(config["batch_size_j"])
            config["batch_size"] = batch_size
            
        if not config["reduceLR_factor"]:
            reduceLR_factor = (config["reduceLR_x"])**2
            config["reduceLR_factor"] = reduceLR_factor
            
        print("DEBUG: ", config)

        print("training func starts")
        print("current work dir = " + os.getcwd())
        deep=RayTune_CellSpike.trainer(config, args)
        print("trainer constructed")

        plot=Plotter_CellSpike(args,deep.metaD )
        deep.init_genetors() # before building model, so data dims are known
        print("init_genetors gets called")

        try:
            deep.build_model()
        except:
            print('M: deep.build_model failed') #probably HPAR were pathological
            raise Exception('Trial failed due to bad hyper parameters')
            #exit(0)

        if args.trainTime >200: deep.save_model_full() # just to get binary dump of the graph

        if args.seedWeights=='same' :  args.seedWeights=args.outPath
        if args.seedWeights:
            deep.load_weights_only(path=args.seedWeights)

        sumF=deep.outPath+deep.prjName+'.sum_train.yaml'
        write_yaml(deep.sumRec, sumF) # to be able to predict while training continus
        
        

        deep.train_model()

        deep.save_model_full()

        try:
            plot.train_history(deep,figId=10)
        except:
            deep.sumRec['plots']='failed'
            print('M: plot.train_history failed')

        write_yaml(deep.sumRec, sumF)

        plot.display_all('train')

    return training_function

#custom sampler functions for search space. The logic for these functions came from genHPar_CellSpike.py

# sampling conv_filter
def get_conv_filter(spec):
    numLyr=np.random.randint(3,8)
    filt1=int(np.random.choice([8,16,32,64,128]))
    filtL=[filt1]
    for i in range(1,numLyr):
        fact=np.random.choice([1,2,4])
        filt1=int(min(256,filt1*fact))
        filtL.append(filt1)

    return filtL

# sampling fc_dims
def get_fc_dims(spec):
    numLyr=np.random.randint(2,8)
    nOut = 31
    filt1=nOut*np.random.choice([1,2,4,8,16])
    #print('oo',nOut,filt1)
    filtL=[int(filt1)]
    for i in range(1,numLyr):
        fact=np.random.choice([1,2,4])
        filt1=int(min(512,filt1*fact))
        filtL.insert(0,filt1)

    return filtL

# generates batch size
def get_batch_size(spec):
    j=np.random.randint(5,8)
    bs=1<<j
    return bs

#generates LR factor
def get_reduceLR(spec):
    xx=np.random.uniform(0.2,0.8)
    lrReduce=xx*xx
    return lrReduce

#form opt list
def get_opt(spec):
    optName = str(np.random.choice(['adam','nadam']))
    return [optName, 0.001, 1.1e-7]



print("Connecting to Ray head @ "+os.environ["ip_head"])
#init(address=os.environ["ip_head"])
ray.init(address='auto', _node_ip_address=os.environ["ip_head"].split(":")[0], _redis_password=os.environ["redis_password"])
print("Connected to Ray")

if args.nodes == "GPU":
    # Using raytune on a Slurm cluster
    print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
    print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print(ray.cluster_resources())



'''
#mutation space (for pbt). NOTE: tune.sample_from is not allowed in the mutation space, so for now I will not put them here.

mutation_config = {'conv_kernel' : tune.randint(3, 10),
          'conv_repeat' : tune.randint(3, 10),
          'pool_len' : tune.randint(1, 4),
          'dropFrac' : tune.choice([0.01, 0.02, 0.05, 0.10]),
          'lossName' : tune.choice(['mse', 'mae']),
          }

#mutation custom explore functions

def mutation_custom_explore(config):
    config['conv_filter'] = tune.sample_from(get_conv_filter)
    config['fc_dims'] = tune.sample_from(get_fc_dims)
    config['optimizer'] = tune.sample_from(get_opt)
    config['batch_size'] = tune.sample_from(get_batch_size)
    config['reduceLR_factor'] = tune.sample_from(get_reduceLR)
    return config


# Instatiating our PBT object. Metric is validation loss and mode is minimize since we are trying to minimize loss.
# NOTE: The hyperparam_mutations dictionary was taken from an example and does not apply to our code. I am still figuring out what
# is the difference between hyperparam_mutations and config since they are both search spaces.

pbt = PopulationBasedTraining(
    time_attr="training_iteration",
    metric="val_loss",
    mode="min",
    hyperparam_mutations=mutation_config,
    custom_explore_fn=mutation_custom_explore
    )
'''
# search space
'''
config = {'myID' : tune.sample_from(lambda spec: 'id1' + ('%.11f'%np.random.uniform())[2:]),
          'inp_batch_norm': True,
          'junk1' : [1],
          'numFeature' : None,
          'conv_filter' : tune.sample_from(get_conv_filter),
          'conv_kernel' : tune.randint(3, 10),
          'conv_repeat' : tune.randint(3, 10),
          'pool_len' : tune.randint(1, 4),
          'fc_dims' : tune.sample_from(get_fc_dims),
          'lastAct' : 'tanh',
          'outAmpl' : 1.2,
          'dropFrac' : tune.choice([0.01, 0.02, 0.05, 0.10]),
          'lossName' : tune.choice(['mse', 'mae']),
          'optimizer' : tune.sample_from(get_opt),
          'batch_size' : tune.sample_from(get_batch_size),
          'reduceLR_factor' : tune.sample_from(get_reduceLR),
          'steps' : None,
          }
'''
config = {'myID' : None,
          'inp_batch_norm': True,
          'junk1' : [1],
          'numFeature' : None,
          'conv_filter' : None,
          "cf_num_layers": tune.quniform(3, 7, 1),
          "filter_1": tune.quniform(3, 7, 1),
          "filter_2": tune.randint(0, 3),
          "filter_3": tune.randint(0, 3),
          "filter_4": tune.randint(0, 3),
          "filter_5": tune.randint(0, 3),
          "filter_6": tune.randint(0, 3),
          "filter_7": tune.randint(0, 3),
          'conv_kernel' : tune.quniform(3, 9, 1),
          'conv_repeat' : tune.quniform(3, 9, 1),
          'pool_len' : tune.quniform(1, 3, 1),
          'fc_dims' : None,
          'fc_num_layers': tune.quniform(2, 7, 1),
          'fc_1': tune.randint(0, 5),
          'fc_2': tune.randint(0, 3),
          'fc_3': tune.randint(0, 3),
          'fc_4': tune.randint(0, 3),
          'fc_5': tune.randint(0, 3),
          'fc_6': tune.randint(0, 3),
          'fc_7': tune.randint(0, 3),
          'lastAct' : 'tanh',
          'outAmpl' : 1.2,
          'dropFrac' : tune.choice([0.01, 0.02, 0.05, 0.10]),
          'lossName' : tune.choice(['mse', 'mae']),
          'optimizer' : None,
          'optName': tune.choice(['adam','nadam']),
          'batch_size' : None,
          'batch_size_j': tune.quniform(5, 7, 1),
          'reduceLR_factor' : None,
          'reduceLR_x' : tune.uniform(0.2, 0.8),
          'steps' : None
         }

# ASHA Scheduler
asha = AsyncHyperBandScheduler(time_attr='training_iteration',
                               metric="val_loss",
                               mode="min",
                               grace_period=6)

# HyperOpt

initial_best_config = [{'conv_filter' : [32, 64, 64, 64, 128, 128],
                        'conv_kernel' : 7,
                        'conv_repeat' : 1,
                        'pool_len' : 3,
                        'fc_dims' : [256, 256, 240],
                        'dropFrac' : 0.01,
                        'lossName' : 'mse',
                        'optimizer' : ['nadam', 0.001, 1.1e-07],
                        'batch_size' : 128,
                        'reduceLR_factor' : 0.448298765780917
                       },
                       {'conv_filter' : [32, 64, 128, 256, 256],
                        'conv_kernel' : 3,
                        'conv_repeat' : 1,
                        'pool_len' : 4,
                        'fc_dims' : [256, 256, 256, 120, 30],
                        'dropFrac' : 0.05,
                        'lossName' : 'mae',
                        'optimizer' : ['nadam', 0.001, 1.1e-07],
                        'batch_size' : 128,
                        'reduceLR_factor' : 0.40318386178240434
                       }
                      ]

initial_best_config2 = [{'myID' : None,
                        'inp_batch_norm': True,
                        'junk1' : [1],
                        'numFeature' : None,
                        'conv_filter' : None,
                        "cf_num_layers": 6.0,
                        "filter_1": 5,
                        "filter_2": 1,
                        "filter_3": 0,
                        "filter_4": 0,
                        "filter_5": 1,
                        "filter_6": 0,
                        "filter_7": 0,
                        'conv_kernel' : 7.0,
                        'conv_repeat' : 1.0,
                        'pool_len' : 3.0,
                        'fc_dims' : None,
                        'fc_num_layers': 3.0,
                        'fc_1': 8,
                        'fc_2': 0,
                        'fc_3': 0,
                        'fc_4': 0,
                        'fc_5': 0,
                        'fc_6': 0,
                        'fc_7': 0,
                        'lastAct' : 'tanh',
                        'outAmpl' : 1.2,
                        'dropFrac' : 0,
                        'lossName' : 0,
                        'optimizer' : None,
                        'optName': 1,
                        'batch_size' : None,
                        'batch_size_j': 7.0,
                        'reduceLR_factor' : None,
                        'reduceLR_x' : 0.66955116741,
                        'steps' : None
                        },
                        {'myID' : None,
                        'inp_batch_norm': True,
                        'junk1' : [1],
                        'numFeature' : None,
                        'conv_filter' : None,
                        "cf_num_layers": 5.0,
                        "filter_1": 5,
                        "filter_2": 2,
                        "filter_3": 1,
                        "filter_4": 1,
                        "filter_5": 0,
                        "filter_6": 0,
                        "filter_7": 0,
                        'conv_kernel' : 3.0,
                        'conv_repeat' : 1.0,
                        'pool_len' : 4.0,
                        'fc_dims' : None,
                        'fc_num_layers': 5.0,
                        'fc_1': 5,
                        'fc_2': 2,
                        'fc_3': 1,
                        'fc_4': 0,
                        'fc_5': 0,
                        'fc_6': 0,
                        'fc_7': 0,
                        'lastAct' : 'tanh',
                        'outAmpl' : 1.2,
                        'dropFrac' : 2,
                        'lossName' : 1,
                        'optimizer' : None,
                        'optName': 1,
                        'batch_size' : None,
                        'batch_size_j': 7.0,
                        'reduceLR_factor' : None,
                        'reduceLR_x' : 0.63496760687,
                        'steps' : None
                        }
                       ]

                  

hyperopt = HyperOptSearch(metric="val_loss",
                          mode="min",
                          points_to_evaluate=initial_best_config2)
# Metric and mode are redundant so RayTune said to remove them from either pbt or tune.run. Num_samples is the number of trials (different hpam combinations?), #
# which is set to 10 for now. Scheduler is the PBT object instatiated above.


if args.nodes == "CPU":
    resources = {"cpu":int(args.numCPU)}
else:
    resources = {"cpu":10,"gpu":int(args.numGPU)}

analysis = tune.run(
    training_initialization(),
    resources_per_trial=resources,
    scheduler=asha,
    search_alg=hyperopt,
    num_samples=int(args.numHparams),
    config=config,
    local_dir = args.rayResult)


"""
python ./train_RayTune.py --dataPath /global/homes/b/balewski/prjn/neuronBBP-pack40kHzDisc/probe_quad/bbp153 --probeType quad -t 60 --useDataFrac 0.05 --rayResult $SCRATCH/ray_results --numHparams 1 --maxEpochTime 4800
"""
