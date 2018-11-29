#!/usr/bin/env python
"""
Animals with Attributes Dataset, http://attributes.kyb.tuebingen.mpg.de
Perform Multiclass Predicition from binary attributes and evaluates it.
(C) 2009 Christoph Lampert <chl@tuebingen.mpg.de>
"""

import os,sys
sys.path.append('/agbs/cluster/chl/libs/python2.5/site-packages/')
from numpy import *

def nameonly(x):
    return x.split('\t')[1]

def loadstr(filename,converter=str):
    return [converter(c.strip()) for c in file(filename).readlines()]

def loaddict(filename,converter=str):
    D={}
    for line in file(filename).readlines():
        line = line.split()
        D[line[0]] = converter(line[1].strip())
    
    return D

# adapt these paths and filenames to match local installation

classnames = loadstr('../classes.txt',nameonly)
numexamples = loaddict('numexamples.txt',int)

def evaluate(split,C):
    global test_classnames
    attributepattern = './DAP/cvfold%d_C%g_%%02d.prob' % (split,C)
    
    if split == 0:
        test_classnames=loadstr('/agbs/share/datasets/Animals_with_Attributes/testclasses.txt')
        train_classnames=loadstr('/agbs/share/datasets/Animals_with_Attributes/trainclasses.txt')
    else:
        startid= (split-1)*10
        stopid = split*10
        test_classnames = classnames[startid:stopid]
        train_classnames = classnames[0:startid]+classnames[stopid:]
    
    test_classes = [ classnames.index(c) for c in test_classnames]
    train_classes = [ classnames.index(c) for c in train_classnames]

    M = loadtxt('/agbs/share/datasets/Animals_with_Attributes/predicate-matrix-binary.txt',dtype=float)

    L=[]
    for c in test_classes:
        L.extend( [c]*numexamples[classnames[c]] )

    L=array(L)  # (n,)

    P = []
    for i in range(85):
        P.append(loadtxt(attributepattern % i,float))

    P = array(P).T   # (85,n)

    prior = mean(M[train_classes],axis=0)
    prior[prior==0.]=0.5
    prior[prior==1.]=0.5    # disallow degenerated priors
    M = M[test_classes] # (10,85)

    prob=[]
    for p in P:
        prob.append( prod(M*p + (1-M)*(1-p),axis=1)/prod(M*prior+(1-M)*(1-prior), axis=1) )

    MCpred = argmax( prob, axis=1 )
    
    d = len(test_classes)
    confusion=zeros([d,d])
    for pl,nl in zip(MCpred,L):
        try:
            gt = test_classes.index(nl)
            confusion[gt,pl] += 1.
        except:
            pass

    for row in confusion:
        row /= sum(row)
    
    return confusion,asarray(prob),L


def plot_confusion(confusion):
    from pylab import figure,imshow,clim,xticks,yticks,axis,setp,gray,colorbar,savefig,gca
    fig=figure(figsize=(10,9))
    imshow(confusion,interpolation='nearest',origin='upper')
    clim(0,1)
    xticks(arange(0,10),[c.replace('+',' ') for c in test_classnames],rotation='vertical',fontsize=24)
    yticks(arange(0,10),[c.replace('+',' ') for c in test_classnames],fontsize=24)
    axis([-.5,9.5,9.5,-.5])
    setp(gca().xaxis.get_major_ticks(), pad=18)
    setp(gca().yaxis.get_major_ticks(), pad=12)
    fig.subplots_adjust(left=0.30)
    fig.subplots_adjust(top=0.98)
    fig.subplots_adjust(right=0.98)
    fig.subplots_adjust(bottom=0.22)
    gray()
    colorbar(shrink=0.79)
    savefig('AwA-ROC-confusion-DAP.pdf')
    return 

def plot_roc(P,GT):
    from pylab import figure,xticks,yticks,axis,setp,gray,colorbar,savefig,gca,clf,plot,legend,xlabel,ylabel
    from roc import roc
    AUC=[]
    CURVE=[]
    for i,c in enumerate(test_classnames):
        class_id = classnames.index(c)
        tp,fp,auc=roc(None,GT==class_id,  P[:,i] ) # larger is better
        print "AUC: %s %5.3f" % (c,auc)
        AUC.append(auc)
        CURVE.append(array([fp,tp]))
    order = argsort(AUC)[::-1]
    styles=['-','-','-','-','-','-','-','--','--','--']
    figure(figsize=(9,5))
    for i in order:
        c = test_classnames[i]
        plot(CURVE[i][0],CURVE[i][1],label='%s (AUC: %3.2f)' % (c,AUC[i]),linewidth=3,linestyle=styles[i])
    
    legend(loc='lower right')
    xticks([0.0,0.2,0.4,0.6,0.8,1.0], [r'$0$', r'$0.2$',r'$0.4$',r'$0.6$',r'$0.8$',r'$1.0$'],fontsize=18)
    yticks([0.0,0.2,0.4,0.6,0.8,1.0], [r'$0$', r'$0.2$',r'$0.4$',r'$0.6$',r'$0.8$',r'$1.0$'],fontsize=18)
    xlabel('false negative rate',fontsize=18)
    ylabel('true positive rate',fontsize=18)
    savefig('AwA-ROC-DAP.pdf')

def main():
    try:
        split = int(sys.argv[1])
    except IndexError:
        split = 0

    try:
        C = float(sys.argv[2])
    except IndexError:
        C = 10.

    confusion,prob,L = evaluate(split,C)
    print "Mean class accuracy %g" % mean(diag(confusion)*100)
    plot_confusion(confusion) 
    plot_roc(prob,L)
    
if __name__ == '__main__':
    main()
