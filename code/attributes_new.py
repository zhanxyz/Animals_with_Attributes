#!/usr/bin/env python
"""
Animals with Attributes Dataset, http://attributes.kyb.tuebingen.mpg.de
Train one binary attribute classifier using all possible features.
Needs "shogun toolbox with python interface" for SVM training
(C) 2009 Christoph Lampert <chl@tuebingen.mpg.de>
updated to new Shogun version: (C) 2011 Christoph Lampert <chl@ist.ac.at> 
"""

import os,sys
sys.path.append('/nfs3group/chlgrp/datasets/Animals_with_Attributes/code/')

from numpy import *
from platt import *

import cPickle, bz2

def nameonly(x):
    return x.split('\t')[1]

def loadstr(filename,converter=str):
    return [converter(c.strip()) for c in file(filename).readlines()]

def bzUnpickle(filename):
    return cPickle.load(bz2.BZ2File(filename))

# adapt these paths and filenames to match local installation

feature_pattern =  '/nfs3group/chlgrp/datasets/Animals_with_Attributes/code/feat/%s-%s.pic.bz2'
labels_pattern =  '/nfs3group/chlgrp/datasets/Animals_with_Attributes/code/feat/%s-labels.pic.bz2'

all_features = ['cq','lss','phog','sift','surf','rgsift']

attribute_matrix = 2*loadtxt('/nfs3group/chlgrp/datasets/Animals_with_Attributes/predicate-matrix-binary.txt',dtype=float)-1
classnames = loadstr('/nfs3group/chlgrp/datasets/Animals_with_Attributes/classes.txt',nameonly)
attributenames = loadstr('/nfs3group/chlgrp/datasets/Animals_with_Attributes/predicates.txt',nameonly)

def create_data(all_classes,attribute_id):
    featurehist={}
    for feature in all_features:
        featurehist[feature]=[]
    
    labels=[]
    for classname in all_classes:
        class_id = classnames.index(classname)
        class_size = 0
        for feature in all_features:
            featurefilename = feature_pattern % (classname,feature)
            print '# ',featurefilename
            histfile = bzUnpickle(featurefilename)
            featurehist[feature].extend( histfile )
        
        labelfilename = labels_pattern % classname
        print '# ',labelfilename
        print '#'
        labels.extend( bzUnpickle(labelfilename)[:,attribute_id] )
    
    for feature in all_features:
        featurehist[feature]=array(featurehist[feature]).T  # shogun likes its data matrices shaped FEATURES x SAMPLES
    
    labels = array(labels)
    return featurehist,labels

def train_attribute(attribute_id, C, split=0):
    from shogun import Classifier,Features,Kernel,Distance
    attribute_id = int(attribute_id)
    print "# attribute ",attributenames[attribute_id]
    C = float(C)
    print "# C ", C
    
    if split == 0:
        train_classes=loadstr('/nfs3group/chlgrp/datasets/Animals_with_Attributes/trainclasses.txt')
        test_classes=loadstr('/nfs3group/chlgrp/datasets/Animals_with_Attributes/testclasses.txt')
    else:
        classnames = loadstr('/nfs3group/chlgrp/datasets/Animals_with_Attributes/classnames.txt')
        startid= (split-1)*10
        stopid = split*10
        test_classes = classnames[startid:stopid]
        train_classes = classnames[0:startid]+classnames[stopid:]
    
    Xtrn,Ltrn = create_data(train_classes,attribute_id)
    Xtst,Ltst = create_data(test_classes,attribute_id)
    
    if min(Ltrn) == max(Ltrn):  # only 1 class
        Lprior = mean(Ltrn)
        prediction = sign(Lprior)*ones(len(Ltst))
        probabilities = 0.1+0.8*0.5*(Lprior+1.)*ones(len(Ltst)) # fallback
        return prediction,probabilities,Ltst
    
    #sg('loglevel', 'WARN')
    widths={}
    for feature in all_features:
        traindata = array(Xtrn[feature][:,::50],float) # used to be 5*offset
        trainfeat = Features.RealFeatures(traindata)
        DM=Distance.ChiSquareDistance(trainfeat,trainfeat).get_distance_matrix()
        widths[feature] = median(DM.flatten())
        del traindata,trainfeat,DM
    
    s = Classifier.LibSVM() #sg('new_svm', 'LIBSVM')
    #sg('use_mkl', False)     # we use fixed weights here
    
    #sg('clean_features', 'TRAIN')
    #sg('clean_features', 'TEST')
    
    Lplatt_trn = concatenate([Ltrn[i::10] for i in range(9)])   # 90% for training
    Lplatt_val = Ltrn[9::10] # remaining 10% for platt scaling 
    
    feats_trn = Features.CombinedFeatures()
    feats_val = Features.CombinedFeatures()
    for feature in all_features:
        Xplatt_trn = concatenate([Xtrn[feature][:,i::10] for i in range(9)], axis=1)
        feats_trn.append_feature_obj( Features.RealFeatures(ascontiguousarray(Xplatt_trn)) )
        #sg('add_features', 'TRAIN', Xplatt_trn)
        Xplatt_val = Xtrn[feature][:,9::10]
        feats_val.append_feature_obj( Features.RealFeatures(ascontiguousarray(Xplatt_val)) )
        #sg('add_features', 'TEST', Xplatt_val)
        del Xplatt_trn,Xplatt_val,Xtrn[feature]
    
    labels_trn = Features.Labels(Lplatt_trn)
    #sg('set_labels', 'TRAIN', Lplatt_trn)
    
    kernel = Kernel.CombinedKernel()
    #sg('set_kernel', 'COMBINED', 5000)
    for featureset in all_features:
        kernel.append_kernel( Kernel.Chi2Kernel( 5000, widths[featureset]/5. ) )
        #sg('add_kernel', 1., 'CHI2', 'REAL', 10, widths[featureset]/5. )
    
    kernel.init(feats_trn,feats_trn)
    K=kernel.get_kernel_matrix()
    K.tofile('/scratch/chl/cvfold%d_C%g_%02d-trn.kernel' %  (split, C, attribute_id))
    del K
    
    s.set_max_train_time(600*60.)
    #sg('svm_max_train_time', 600*60.) # one hour should be plenty
    s.set_C(C,C)
    #sg('c', C)
    
    s.set_kernel(kernel)
    s.set_labels(labels_trn)
    #sg('init_kernel', 'TRAIN')
    try:
        s.train()
        #sg('train_classifier')
    except (RuntimeWarning,RuntimeError):    # can't train, e.g. all samples have the same labels
        Lprior = mean(Ltrn)
        prediction = sign(Lprior) * ones(len(Ltst))
        probabilities = 0.1+0.8*0.5*(Lprior+1.) * ones(len(Ltst))
        savetxt('./DAP/cvfold%d_C%g_%02d.txt' % (split, C, attribute_id), prediction)
        savetxt('./DAP/cvfold%d_C%g_%02d.prob' % (split, C, attribute_id), probabilities)
        savetxt('./DAP/cvfold%d_C%g_%02d.labels' % (split, C, attribute_id), Ltst)
        return prediction,probabilities,Ltst
    
    bias = s.get_bias()
    alphas = s.get_alphas()
    #[bias, alphas]=sg('get_svm')
    #print bias,alphas
    
    kernel.init(feats_trn,feats_val)
    K=kernel.get_kernel_matrix()
    K.tofile('/scratch/chl/cvfold%d_C%g_%02d-val.kernel' %  (split, C, attribute_id))
    del K

    #sg('init_kernel', 'TEST')
    try:
        prediction=s.classify().get_labels()
        #prediction=sg('classify')
        platt_params = SigmoidTrain(prediction, Lplatt_val)
        probabilities = SigmoidPredict(prediction, platt_params)
        
        savetxt('./DAP/cvfold%d_C%g_%02d-val.txt' % (split, C, attribute_id), prediction)
        savetxt('./DAP/cvfold%d_C%g_%02d-val.prob' % (split, C, attribute_id), probabilities)
        savetxt('./DAP/cvfold%d_C%g_%02d-val.labels' % (split, C, attribute_id), Lplatt_val)
        savetxt('./DAP/cvfold%d_C%g_%02d-val.platt' % (split, C, attribute_id), platt_params)
        #print '#train-perf ',attribute_id,C,mean((prediction*Lplatt_val)>0),mean(Lplatt_val>0)
        #print '#platt-perf ',attribute_id,C,mean((sign(probabilities-0.5)*Lplatt_val)>0),mean(Lplatt_val>0)
    except RuntimeError:
        Lprior = mean(Ltrn)
        prediction = sign(Lprior)*ones(len(Ltst))
        probabilities = 0.1+0.8*0.5*(Lprior+1.)*ones(len(Ltst))
        print >> sys.stderr, "#Error during testing. Using constant platt scaling"
        platt_params=[1.,0.]
    
    # ----------------------------- now apply to test classes ------------------
    
    feats_tst = Features.CombinedFeatures()
    #sg('clean_features', 'TEST')
    for feature in all_features:
        feats_tst.append_feature_obj( Features.RealFeatures(ascontiguousarray(Xtst[feature])) )
        del Xtst[feature]
    
    kernel.init(feats_trn,feats_tst)
    K=kernel.get_kernel_matrix()
    K.tofile('/scratch/chl/cvfold%d_C%g_%02d-tst.kernel' %  (split, C, attribute_id))
    del K

    #sg('init_kernel', 'TEST')
    prediction=s.classify().get_labels()
    #prediction=sg('classify')
    probabilities = SigmoidPredict(prediction, platt_params)
    
    savetxt('./DAP/cvfold%d_C%g_%02d.txt' % (split, C, attribute_id), prediction)
    savetxt('./DAP/cvfold%d_C%g_%02d.prob' % (split, C, attribute_id), probabilities)
    savetxt('./DAP/cvfold%d_C%g_%02d.labels' % (split, C, attribute_id), Ltst)
    
    #print '#test-perf ',attribute_id,C,mean((prediction*Ltst)>0),mean(Ltst>0)
    #print '#platt-perf ',attribute_id,C,mean((sign(probabilities-0.5)*Ltst)>0),mean(Ltst>0)
    return prediction,probabilities,Ltst

if __name__ == '__main__':
    import sys

    try:
        attribute_id = int(sys.argv[1])
    except IndexError:
        print "Must specify attribute ID!"
        raise SystemExit
    try:
        split = int(sys.argv[2])
    except IndexError:
        split = 0
    try:
        C = float(sys.argv[3])
    except IndexError:
        C = 10.

    pred,prob,Ltst = train_attribute(attribute_id,C,split)
    print "Done.", attribute_id, C, split
