clear all, close all

% dataset
pnam = '/agbs/share/datasets/Animals_with_Attributes';
% output
outpath = '.';

% There are 6 feature representations:
% - cq: (global) color histogram (1x1 + 2x2 + 4x4 spatial pyramid, 128 bins each, each histogram L1-normalized)
% - lss[1]: local self similarity (2000 entry codebook, raw bag-of-visual-word counts)
% - phog[2]: histogram of oriented gradients (1x1 + 2x2 + 4x4 spatial pyramid, 12 bins each, each histogram L1-normalized or all zero)
% - rgsift[3]: rgSIFT descriptors (2000 entry codebook, bag-of-visual-word counts, L1-normalized)
% - sift[4]: SIFT descriptors (2000 entry codebook, raw bag-of-visual-word counts)
% - surf[5]: SUFT descriptors (2000 entry codebook, raw bag-of-visual-word counts)
feat = {'cq','lss','phog','rgsift','sift','surf'};
nfeat = [2688,2000,252,2000,2000,2000];
% [1] E. Shechtman, and M. Irani: "Matching Local Self-Similarities 
%     across Images and Videos", CVPR 2007.
% 
% [2] A. Bosch, A. Zisserman, and X. Munoz: "Representing shape with 
%     a spatial pyramid kernel", CIVR 2007.
% 
% [3] Koen E. A. van de Sande, Theo Gevers and Cees G. M. Snoek:
%     "Evaluation of Color Descriptors for Object and Scene 
%     Recognition", CVPR 2008.
% 
% [4] D. G. Lowe, "Distinctive Image Features from Scale-Invariant 
%     Keypoints", IJCV 2004.
%     
% [5] H. Bay, T. Tuytelaars, and L. Van Gool: "SURF: Speeded Up 
%     Robust Features", ECCV 2006.

%% set some constants
% class names of all classes
[tmp,classes] = textread([pnam,'/classes.txt'],'%d %s'); clear tmp
% class names of training/test classes
trainclasses  = textread([pnam,'/trainclasses.txt'],'%s');
testclasses   = textread([pnam,'/testclasses.txt' ],'%s');
% classes(trainclasses_id) == trainclasses
trainclasses_id = -ones(length(trainclasses),1);
for i=1:length(trainclasses)
    for j=1:length(classes)
        if strcmp(trainclasses{i},classes{j})
            trainclasses_id(i) = j;
        end
    end
end
% classes(testclasses_id) == testclasses
testclasses_id = -ones(length(testclasses),1);
for i=1:length(testclasses)
    for j=1:length(classes)
        if strcmp(testclasses{i},classes{j})
            testclasses_id(i) = j;
        end
    end
end

% predicate names of all 85 predicates
[tmp,predicates] = textread([pnam,'/predicates.txt'],'%d %s');
% pca matrix: probability class-attribute pca(i,j) =  P(a_j=1|c=i)
% contains RELATIVE CONNECTION STRENGTH linearly scaled to 0..100
pca = textread([pnam,'/predicate-matrix-continuous.txt']); 
% class antelope has 4 missing values (black,white,blue,brown) => copy from lion
pca(1,1:4) = pca(43,1:4);
% derive binary matrix from continuous
pca_bin = pca > mean(pca(:)); 
% pca_bin = textread([pnam,'/predicate-matrix-binary.txt']); 
save([outpath,'/constants.mat'],'pnam','feat','nfeat','classes',...
    'trainclasses','testclasses','trainclasses_id','testclasses_id', ...
    'predicates','pca','pca_bin')


%% save Matlab files one per feature type
nperclass = zeros(length(classes),1);
for idc = 1:50
    for idf = [1:2,4:6]
        fnam = [pnam,'/Features/',feat{idf},'-hist/',classes{idc}];
        no = numel(dir(fnam))-2;
        nperclass(idc) = no;

        Xc = sparse(nfeat(idf),no);
        for ido = 1:no
            Xc(:,ido) = textread(sprintf('%s/%s_%04d.txt',fnam,classes{idc},ido),'%f');
        end
        fprintf('%s\t%04d: %s\n',feat{idf},ido,classes{idc})
        save(sprintf('%s/feat/x_%s_c%02d.mat',outpath,feat{idf},idc),'Xc')
    end
end
save([outpath,'/nperclass.mat'],'nperclass')
