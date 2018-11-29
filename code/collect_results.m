datapath = '.';
load([datapath,'/constants.mat'])

for cvsplit = 0:5 % 1:5
    for log3_C = -13:-9 % -13:-9
        fnam = sprintf('%s/cv/liblinear_cvfold%d_l3C%d.mat',datapath,cvsplit,log3_C);   
        if exist(fnam,'file')
            data = load(fnam);
            
            % recompute predictions
            % calculate p( attribute  = j | image ) from p( train class = j | image )
            pfa_te = data.pfc_te * ( pca ./ repmat(sum(pca,2),1,85) );
            % calculate p( test class = j | image ) from p( attribute   = j | image )
            s_pcate = sum(pca(data.cte,:));
            is_pcate = zeros(size(s_pcate));  
            is_pcate(s_pcate~=0) = 1./s_pcate(s_pcate~=0); 
            pfc_pr = pfa_te * (pca(data.cte,:).*repmat(is_pcate,10,1))';
            % class assignment
            mx = repmat( max(pfc_pr,[],2), [1,size(pfc_pr,2)] ) == pfc_pr;
            id = 1:size(mx,2); ypr = zeros(size(mx,1),1);
            for i=1:length(ypr)
                if sum(mx(i,:))==0, mx(i,1)=1; end % default is first test class
                ypr(i) = data.cte( id( mx(i,:) ) );  
            end
            acc_pr = 100*sum(ypr==data.yte)/numel(ypr);
            fprintf('split %d, C=%1.2e: Acc = %1.3f%% (%d/%d)\n',...
                cvsplit,3^log3_C,acc_pr,sum(ypr==data.yte),numel(ypr))
        else
            fprintf('%s missing\n',fnam)
        end
    end
end
