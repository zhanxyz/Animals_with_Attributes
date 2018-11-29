function liblinear_cv5(cvsplit,log3_C)

% path to liblinear
addpath /agbs/cluster/hn/mpi_animal_challenge/lib/liblinear-1.33/matlab
% path to Matlab feature representation
datapath = '/kyb/agbs/chl/mysrc/Animals-with-Attributes/code';

% build training-testing split
if cvsplit==0
    % get original split
    tmp = load([datapath,'/constants.mat'],'trainclasses_id','testclasses_id');
    cte = tmp.testclasses_id';
    ctr = tmp.trainclasses_id';
    clear tmp
else
    % build training-testing split
    cte = (cvsplit-1)*10+(1:10); % test classes
    ctr = setdiff(1:50,cte);     % training classes
end
load([datapath,'/constants.mat'])

%% load training data (40 classes)
fprintf('Load training set\n') 
Xtr = []; ytr = [];
for idc = ctr % 40 classes
    Xc = [];
    for idf = 1:6 % 6 features       
        data = load(sprintf('%s/feat/x_%s_c%02d.mat',datapath,feat{idf},idc),'Xc');
        Xc = [Xc; data.Xc];       
    end
    Xtr = [Xtr,Xc];
    ytr = [ytr; idc*ones(size(Xc,2),1)];
    fprintf('  %s(%d)\n',classes{idc},size(Xc,2))
end, Xtr = Xtr';

% train model
fprintf('Learning\n')   
% logistic regression
C = 3^log3_C;
argstr = sprintf('-s 0 -c %f',C);
model  = train(ytr, Xtr, argstr);

%% make prediction on training data
tic
[l,acc_tr,p]  = predict(ytr, Xtr, model, '-b 1');
T = toc;
fprintf('training took %1.2f s\n',T)
pfc_tr = zeros(length(l),50); pfc_tr(:,model.Label) = p; % full 50 matrix

%% load test data (10 classes)
fprintf('Load test set\n') 
Xte = []; yte = [];
for idc = cte % 10 classes
    Xc = [];
    for idf = 1:6 % 6 features
        data = load(sprintf('%s/feat/x_%s_c%02d.mat',datapath,feat{idf},idc),'Xc');
        Xc = [Xc; data.Xc];       
    end
    Xte = [Xte,Xc];
    yte = [yte; idc*ones(size(Xc,2),1)];
    fprintf('  %s(%d)\n',classes{idc},size(Xc,2))
end, Xte = Xte';

%% predict train classes on test data
[l,acc_te,p]  = predict(yte, Xte, model, '-b 1');
pfc_te = zeros(length(l),50); pfc_te(:,model.Label) = p; % full 50 matrix

%% predict test classes on test data
% calculate p( attribute  = j | image ) from p( train class = j | image )
pfa_te = pfc_te * ( prca ./ repmat(sum(prca,2),1,85) );
% calculate p( test class = j | image ) from p( attribute   = j | image )
pfc_pr = pfa_te * (prca(cte,:)./repmat(sum(prca(cte,:)),10,1))';

% class assignment
mx = repmat( max(pfc_pr,[],2), [1,size(pfc_pr,2)] ) == pfc_pr;
id = 1:size(mx,2); ypr = zeros(size(mx,1),1);
for i=1:length(ypr)
    if sum(mx(i,:))==0, mx(i,1)=1; end % default is first test class
    ypr(i) = cte( id( mx(i,:) ) );  
end
acc_pr = 100*sum(ypr==yte)/numel(ypr);
fprintf('Accuracy = %1.4f%% (%d/%d)\n',acc_pr,sum(ypr==yte),numel(ypr))

% save results
fnam = sprintf('%s/cv/liblinear_cvfold%d_l3C%d.mat',datapath,cvsplit,log3_C);
save(fnam,'cvsplit','log3_C','argstr','C','acc_tr','acc_pr',...
    'ctr','cte','pfc_tr','pfc_te','pfc_pr','ytr','yte','ypr')
