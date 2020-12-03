__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import os, time
import warnings
import socket  # for hostname
import copy

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

print('deep-libs imported TF ver:',tf.__version__,' elaT=%.1f sec,'%(time.time() - startT0))


#............................
#............................
#............................
class Deep_CellSpike(object):

    def __init__(self,**kwargs):
        for k, v in kwargs.items():
            self.__setattr__(k, v)
        
        for xx in [ self.dataPath, self.outPath]:
            if os.path.exists(xx): continue
            print('Aborting on start, missing  dir:',xx)
            exit(99)
        
        self.metaF='/meta.cellSpike_%s.yaml'%self.probeType
        print(self.__class__.__name__,'TF ver:', tf.__version__,', prj:',self.prjName)

        # - - - - - - - - - data containers - - - - - - - -
        #self.data={} # data divided by train/val/test or by seg=0,1,...19


    # alternative constructors

    @classmethod #........................
    def trainer(cls, args):
        print('Cnst:train')
        obj=cls(**vars(args))
        obj.read_metaInp(obj.dataPath+obj.metaF)
        obj.train_hirD={'acc': [],'loss': [],'lr': [],'val_acc': [],'val_loss': []}

        # . . . load hyperparameters
        #hparF='./hpar_'+obj.prjName2+'.yaml'
        hparF= obj.hparPath
        bulk=read_yaml(hparF)
        print('bulk',bulk)
        obj.hparams=bulk

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

    @classmethod #........................
    def predictor(cls, args):
        print('Cnst:sim-pred')
        obj=cls(**vars(args))
        obj.read_metaInp(obj.dataPath+obj.metaF)
        return obj

    ''' # never used
    @classmethod #........................
    def hparValidator(cls,args):
        print('Cnst:hpar-validator')
        args.dataPath='./'
        args.probeType='fake123'
        args.prjName='hpar-validator'
        obj=cls(**vars(args))
        return obj
    '''

#............................
    def read_metaInp(self,metaF):
        bulk=read_yaml(metaF)
        self.metaD=bulk['dataInfo']
        self.metaDrw=bulk['rawInfo'] # only for plotting of some raw info
        print('read metaInp:',self.metaD.keys())
        self.metaD['metaFile']=self.metaF
        if self.verb>1: print('read metaInp dump',self.metaD)
        # update prjName if not 'format'
        self.prjName2=self.prjName+'_'+self.modelDesign


#............................
    def build_model(self):
        hpar=self.hparams
        pr=self.verb>1

        print('build_model hpar:',hpar,pr)

        useBN=hpar['inp_batch_norm']
        if 'dataNormalized' in self.metaD:
             useBN=not self.metaD['dataNormalized']

        #print('gg',useBN, hpar['inp_batch_norm'],  self.metaD['dataNormalized'])
        #xx67
        time_dim=self.metaD['numTimeBin']
        out_dim=self.metaD['numPar']

        [d1,rgb_dim]=self.sumRec['XY_shapes']['X']
        if pr: print('XY_shapes:',self.sumRec['XY_shapes'])

        assert time_dim==d1
        assert [out_dim]==self.sumRec['XY_shapes']['Y']

        # CNN params

        cnn_ker=hpar['conv_kernel']
        # hpar.pool_len -  how much time_bins get reduced per pooling
        # hpar.conv_repeat - how many times repeat CNN before maxPool

        # FC params
        dropFrac=hpar['dropFrac']

        # - - - Assembling model 
        xa = Input(shape=(time_dim,rgb_dim),name='trace_%d'%time_dim)
        print('build_model_cnn inpA:',xa.get_shape(), ' dropFrac=',dropFrac)
        h=xa
        if pr: print("xa :",h.get_shape())
        #print('xa dtype:',tf.keras.backend.dtype(h))

        #h = tf.keras.backend.cast(h, dtype='float32') # not working yet
        #print('h-cast dtype:',tf.keras.backend.dtype(h))

        if useBN:
            h=BatchNormalization(name='inp_BN_cast')(h)
            if pr: print("BN out:",h.get_shape())
        #print('cnn-inp dtype:',tf.keras.backend.dtype(h))

        # .....  CNN-1D layers
        k=0
        for dim in hpar['conv_filter']:
            for j in range(hpar['conv_repeat']):
                k+=1
                h= Conv1D(dim,cnn_ker,activation='linear',padding='valid' ,name='cnn%d_d%d_k%d'%(k,dim,cnn_ker))(h)
                if pr: print("CNN k=%d name=%s out:"%(k,h.name),h.get_shape())
                h =LeakyReLU(name='aca%d'%(k))(h)
            h= MaxPool1D(pool_size=hpar['pool_len'], name='pool_%d'%(k))(h)


        h=Flatten(name='to_1d')(h)
        if pr: print("flatten name=%s out:"%(k,h.name),h.get_shape())
        flattenSize=int(h.get_shape()[1])
        if dropFrac>0:  h = Dropout(dropFrac,name='fdrop')(h)

        # .... FC  layers   
        for i,dim in enumerate(hpar['fc_dims']):
            h = Dense(dim,activation='linear',name='fc%d'%i)(h)
            if pr: print("FC i=%d name=%s out:"%(i,h.name),h.get_shape())
            h =LeakyReLU(name='acb%d'%(i))(h)
            if dropFrac>0:  h = Dropout(dropFrac,name='drop%d'%i)(h)

        y= Dense(out_dim, activation=hpar['lastAct'],name='out_%s'%hpar['lastAct'])(h)
        if pr: print("last FC name=%s out:"%(k,h.name),h.get_shape())
        if hpar['lastAct']=='tanh':
            y = Lambda(lambda val: val*hpar['outAmpl'], name='scaleAmpl_%.1f'%hpar['outAmpl'])(y)

        print('build_model: loss=',hpar['lossName'],' opt=',hpar['optimizer'],' out:',y.get_shape())
        # full model
        model = Model(inputs=xa, outputs=y)

        [optName, initLR, optPar2]=hpar['optimizer']

        if optName=='adam' :
            opt=tf.keras.optimizers.Adam(lr=initLR, epsilon=optPar2)
        elif optName=='nadam' :
            opt=tf.keras.optimizers.Nadam(lr=initLR, epsilon=optPar2)
        elif optName=='adadelta' :
            opt = tf.keras.optimizers.Adadelta(lr=initLR,rho=optPar2)
        else:
            crash_21
        
        model.compile(optimizer=opt, loss=hpar['lossName'])
        self.model=model
        # - - -  Model assembled and compiled        
        print('\nSummary   layers=%d , params=%.1f K, inputs:'%(len(model.layers),model.count_params()/1000.),model.input_shape)
        model.summary() # will print

        # append summary record
        rec={'hyperParams':hpar,
             'modelWeightCnt':int(model.count_params()),
             'modelLayerCnt':len(model.layers),
             'flattenSize' : flattenSize,
             'modelDesign' : self.modelDesign,
             'featureType' : self.metaD['featureType'],
             'hostName' : socket.gethostname(),
             'jobId': self.jobId
        }
        self.sumRec.update(rec)

        
#............................
    def init_genetors(self):
        hpar=self.hparams
        #print('zz',hpar['numFeature'],self.metaD['numFeature'])

        if self.verb: print('init Generators, inpDir=%s, batch_size=%d, steps='%(self.dataPath, hpar['batch_size']),self.steps)

        genConf={'h5nameTemplate':self.metaD['h5nameTemplate'], 'dataPath':self.dataPath,
                 'shuffle':self.shuffle_data,'x_y_aux':False,'useDataFrac':self.useDataFrac}
        
        for x in ['batch_size','numFeature']:
            genConf[x]=hpar[x]
            
        self.sumRec['inpGen']={}
        self.inpGenD={}
        for dom in ['train','val']:
            fakeSteps=hpar['steps']
            if fakeSteps!=None and dom=='val' :   fakeSteps=max(1,fakeSteps//8)
            genConf['fakeSteps']=fakeSteps
            genConf['fileIdxL']= copy.deepcopy(self.metaD['splitIdx'][dom])
            genConf['name']=dom    
            self.inpGenD[dom]=CellSpike_input_generator(genConf,verb=1)
            self.sumRec['inpGen'][dom]=copy.deepcopy(genConf)
        self.sumRec['state']='model_build'
        self.sumRec['XY_shapes']=self.inpGenD[dom].XY_shapes()
        
        if type(hpar['numFeature'])==type(list()):
            self.sumRec['featureByName']=[ self.metaD['featureName'][i] for i in hpar['numFeature']]

#............................
    def train_model(self):
        if self.verb==0: print('train silently ')

        callbacks_list = []
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

    #............................
    def save_model_full(self):
        model=self.model
        fname=self.outPath+'/'+self.prjName+'.model.h5'     
        print('save model  to',fname)
        model.save(fname)


   #............................
    def load_model_full(self,path='',verb=1):
        fname=path+'/'+self.prjName+'.model.h5'     
        print('load model from ',fname)
        self.model=load_model(fname) # creates model from HDF5
        if verb:  self.model.summary()


    #............................
    def load_weights_only(self,path='.'):
        start = time.time()
        inpF5m=path+'/'+self.prjName+'.weights_best.h5'  #_best
        print('load  weights  from',inpF5m,end='... ')
        self.model.load_weights(inpF5m) # restores weights from HDF5
        print('loaded, elaT=%.2f sec'%(time.time() - start))


