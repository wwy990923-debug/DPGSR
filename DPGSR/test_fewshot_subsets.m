% test_fewshot_subsets.m — 每类从Sub取≤6张训练, 全量测试
clc; rng(1);
load('pcayaleb50.mat'); data = double(dataset);
K=38; N=64; l1=1e-1; l2=1e-2;

subset_ranges = {1:7, 8:19, 20:31, 32:45, 46:64};
subset_names  = {'Sub1','Sub2','Sub3','Sub4','Sub5'};
nt_vals = [2,3,4,5,6];

fprintf('Train: ≤6 imgs from ONE subset per class\n');
fprintf('Test:  ALL remaining imgs from ALL subsets\n\n');

fprintf('%-6s %3s %8s %8s %8s %8s %8s %8s\n',...
    'Train','nt','FwdAcc','RevAcc','FusAcc','Gain','nFWD','nREV');
fprintf('%-6s %3s %8s %8s %8s %8s %8s %8s\n',...
    '----','--','------','------','------','----','----','----');

for si=1:5
    tr_src = subset_ranges{si};  % 这个Sub的所有图
    for ni=1:length(nt_vals)
        nt=nt_vals(ni);
        if nt>length(tr_src); continue; end
        tr=tr_src(1:nt);  % 从该Sub取前nt张做训练
        te=setdiff(1:64,tr);  % 其余全部做测试

        S=[];Sl=[];T=[];Tl=[];
        for c=1:K
            for i=tr; S=[S data(:,(c-1)*N+i)]; Sl=[Sl c]; end
            for i=te; T=[T data(:,(c-1)*N+i)]; Tl=[Tl c]; end
        end
        M=size(T,2);

        [~,~,o]=DPGSR_Fusion(S,T,Sl,Tl,nt,l1,l2,'adaptive',[],[],0.65,0.05);
        [~,pf]=max(o.PF,[],1);[~,pr]=max(o.PR,[],1);
        fwd=sum(o.classes(pf)==Tl)/M; rev=sum(o.classes(pr)==Tl)/M;

        Z=o.D; n_fwd=0;n_rev=0;Sh=zeros(K,M);
        for j=1:M
            vf=norm(Z(j,Sl==o.classes(pf(j))),2)/(norm(Z(j,:),2)+1e-12);
            PRm=sort(o.PR(:,j),'descend');PRm=PRm(1)-PRm(2);
            if vf>0.65;alfa=0;n_fwd=n_fwd+1;
            elseif PRm>0.05;alfa=1;n_rev=n_rev+1;
            else alfa=0;n_fwd=n_fwd+1;end
            Sh(:,j)=(1-alfa)*o.PF(:,j)+alfa*o.PR(:,j);
        end
        [~,ph]=max(Sh,[],1); fus=sum(o.classes(ph)==Tl)/M;
        fprintf('%-6s %3d %8.4f %8.4f %8.4f %+8.4f %8d %8d\n',...
            subset_names{si},nt,fwd,rev,fus,fus-fwd,n_fwd,n_rev);
    end
    fprintf('\n');
end
