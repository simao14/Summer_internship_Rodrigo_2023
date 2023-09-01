import argparse
import uproot
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pandas as pd
import numpy as np
import os.path
from sklearn import metrics


def replacespecial(stages):

    string=str(stages[0])
    for i in range(1,len(stages)):
        string+= "_"+ str(stages[i]) 
    return string

def varset(stages):
    
    var=["Btrk1Pt","Btrk2Pt","Trk1DCAz","Trk2DCAz","Trk1DCAxy","Trk2DCAxy","MassDis","dls","Balpha","dls2D","cos(Bdtheta)","Bchi2cl","Btrk1Eta","Btrk2Eta","Bmass","Bpt"]
    stage=[]
    for i in range(len(stages)):
        stage.append(var[stages[i]])
    return stage 

def prepdata(fileS,fileB,ptmin, ptmax):
    
    treeS = fileS["ntKp"]
    treeB = fileB["ntKp"]
    
    signal = treeS.arrays(library="pd")
    background = treeB.arrays(library="pd")

    signal["Trk1DCAz"] = signal.Btrk1Dz1/signal.Btrk1DzError1
    signal["Trk2DCAz"] = signal.Btrk2Dz1/signal.Btrk2DzError1
    signal["Trk1DCAxy"] = signal.Btrk1Dxy1/signal.Btrk1DxyError1
    signal["Trk2DCAxy"] = signal.Btrk2Dxy1/signal.Btrk2DxyError1
    signal["MassDis"] = signal.Btktkmass-1.019455
    signal["dls"] = signal.BsvpvDistance/signal.BsvpvDisErr
    signal["dls2D"] = signal.Bd0
    signal["tag"] = np.ones(signal.shape[0])
    
    background["Trk1DCAz"] = background.Btrk1Dz1/background.Btrk1DzError1
    background["Trk2DCAz"] = background.Btrk2Dz1/background.Btrk2DzError1
    background["Trk1DCAxy"] = background.Btrk1Dxy1/background.Btrk1DxyError1
    background["Trk2DCAxy"] = background.Btrk2Dxy1/background.Btrk2DxyError1
    background["MassDis"] = background.Btktkmass-1.019455
    background["dls"] = background.BsvpvDistance/background.BsvpvDisErr
    background["dls2D"] = background.Bd0
    background["tag"] = np.zeros(background.shape[0])
    
    
    cutS = ( (signal.Bgen==23333) & (signal.Bpt>ptmin) & (signal.Bpt<ptmax) )
    cutB = ( (((background.Bmass - 5.27929 ) > 0.25) &  ((background.Bmass - 5.27929) < 0.30)) & (background.Bpt>ptmin) & (background.Bpt<ptmax) )
    
    signal_cut=signal[cutS]
    background_cut=background[cutB]

    return signal_cut,background_cut

def plot(epochs, plottable, xlabel="Epochs", ylabel='', name='',label=None):
    plt.clf()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(epochs, plottable, label=label)
    plt.legend()
    plt.grid(visible=True)
    plt.savefig('results/%s.pdf' % (name), bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ptmin', type=float)
    parser.add_argument('ptmax', type=float)
    parser.add_argument('-iter', default=20 , type=int)
    parser.add_argument('-lamb', default=1e-06, type=float)
    parser.add_argument('-alpha', default= 1e-06, type=float)
    parser.add_argument('-max_depth', default=3, type=int)
    parser.add_argument('-eta', type=float, default=0.5)
    parser.add_argument('-gamma', type=float, default= 0.07)
    parser.add_argument('-stages', type=int, nargs='+', default=[0,2,4,7,8,9,12])
    parser.add_argument('-booster',
                        choices=['gbtree', 'dart'], default='gbtree')
    parser.add_argument('-grow_policy',
                        choices= ["depthwise", "lossguide"], default='lossguide')
    opt = parser.parse_args()
    
    print("files")
    fileS = uproot.open("/lstore/cms/simao/sample/BPMC_3_60_small2.root")
    fileB = uproot.open("/lstore/cms/simao/sample/BPData_3_60_small2.root")
    #fileS=uproot.open("~/Desktop/UNI/LIP/mnt/data/BPMC_3_60.root")
    #fileB=uproot.open("~/Desktop/UNI/LIP/mnt/data/BPData_3_60.root")
    
    signal,background = prepdata(fileS,fileB,opt.ptmin,opt.ptmax)
    
    stage=varset(opt.stages)
    signal_var=signal[stage]
    background_var=background[stage]

    fullstages=[0,2,4,7,8,9,11,12,14,15]
    fullstagename=varset(fullstages)
    
    x=pd.concat([signal_var,background_var],axis=0,ignore_index=True)
    
    y=pd.concat([signal["tag"],background["tag"]],ignore_index=True)

    train_x, test_x, train_y, test_y = train_test_split(x.to_numpy(), y.to_numpy(), test_size=0.2)
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2)
    
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dvalid = xgb.DMatrix(valid_x, label=valid_y)
    dtest = xgb.DMatrix(test_x, label=test_y)

    # initialize the model

    param = {
        "verbosity": 1,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "booster": opt.booster,
        "lambda": opt.lamb,
        "alpha": opt.alpha,
        "max_depth": opt.max_depth,
        "eta": opt.eta,
        "gamma": opt.gamma,
        "grow_policy": opt.grow_policy,
    }
    

    bst = xgb.train(param, dtrain, opt.iter , evals=[(dvalid, "validation")])

    stagelist=replacespecial(opt.stages)
    config = "BDT-{}-{}-{}".format(opt.ptmin, opt.ptmax,stagelist)

    if not os.path.exists("results/%s" % config):
        os.makedirs("results/%s" % config)
    
    yscore = bst.predict(dtest)
    nn_fpr, nn_tpr, _ = metrics.roc_curve(test_y, yscore)
    auc=metrics.auc(nn_tpr,1-nn_fpr)
    plot(nn_tpr,1-nn_fpr,ylabel="background efficiency", xlabel="signal efficiency",name='{}/ROC-curve'.format(config),label=f'auc:{auc:.3f};')


if __name__ == '__main__':
    main()
    print("Done!")