#!/usr/bin/env python
"""
Animals with Attributes Dataset, http://attributes.kyb.tuebingen.mpg.de
Convert Animals with Attributes features into Python Pickles
(C) 2009 Christoph Lampert <chl@tuebingen.mpg.de>
"""
import sys,os
import cPickle, bz2
from numpy import *

def nameonly(x):
    return x.split('\t')[1]

def loadstr(filename,converter=str):
    return [converter(c.strip()) for c in file(filename).readlines()]

# adapt these paths and filenames to match local installation

all_classnames = loadstr('/agbs/share/datasets/Animals_with_Attributes/classes.txt',nameonly)
attribute_matrix = 2*loadtxt('/agbs/share/datasets/Animals_with_Attributes/predicate-matrix-binary.txt',dtype=float)-1
featurepath_pattern = '/agbs/share/datasets/Animals_with_Attributes/Features/%s-hist/%s/'
outputhist_pattern =  './feat/%s-%s.pic.bz2'
outputlabels_pattern =  './feat/%s-labels.pic.bz2'

def bzPickle(obj,filename):
    f = bz2.BZ2File(filename, 'wb')
    cPickle.dump(obj, f)
    f.close()

def preprocess(classname, all_featurenames=[]):
    if len(all_featurenames)==0:
        all_featurenames = ['cq','lss','phog','sift','surf','rgsift']
    
    labels=[]
    featurehist={}
    for feature in all_featurenames:
        featurehist[feature]=[]
    
    class_id = all_classnames.index(classname)
    class_size = 0
    for feature in all_featurenames:
        featurepath = featurepath_pattern % (feature,classname)
        print '# Reading from ',featurepath,
        histfile = loadtxt(os.popen('cat %s/*.txt' % featurepath))
        if len(histfile.shape)==1:
            histfile.shape=(1,-1)
        class_size = len(histfile)
        
        picklefile = outputhist_pattern % (classname,feature)
        print "Pickling ",classname, " feature ",feature, " to ",picklefile
        bzPickle(histfile, picklefile)
    
    labels = reshape(attribute_matrix[class_id].repeat(class_size,0),(-1,class_size)).T
    picklefile = outputlabels_pattern % classname
    print "Pickling ",classname, " labels to ",picklefile
    bzPickle(labels, picklefile)

if __name__ == '__main__':
    for classname in all_classnames:
        preprocess(classname)

