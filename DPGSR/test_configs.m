% test_configs.m — 固定策略(θ=0.65, m₀=0.05, λ₁=1e-1, λ₂=1e-2)
% 改变 K、nTrain、nTest 看泛化性
clc; rng(1);
load('pcayaleb50.mat'); data = double(dataset);

configs = [
    15,  4, 60;    % K=15, nt=4,  nTest=60
    15,  8, 56;    % K=15, nt=8,  nTest=56 (原始)
    15, 12, 52;    % K=15, nt=12, nTest=52
    15, 16, 48;    % K=15, nt=16, nTest=48
    20,  8, 56;    % K=20, nt=8,  nTest=56
    25,  8, 56;    % K=25, nt=8,  nTest=56
    30,  8, 56;    % K=30, nt=8,  nTest=56
    38,  8, 56;    % K=38, nt=8,  nTest=56
];
nSplits = 5;

fprintf('Fixed: verif hard-switch (th=0.65, m=0.05), l1=1e-1, l2=1e-2\n\n');
fprintf('%-4s %-4s %-4s %8s %8s %8s %8s %8s %8s\n', ...
    'K', 'nt', 'nT', 'FwdAcc', 'RevAcc', 'FusAcc', 'Broken', 'Fixed', 'Net');
fprintf('%-4s %-4s %-4s %8s %8s %8s %8s %8s %8s\n', ...
    '---', '---', '---', '------', '------', '------', '------', '-----', '---');

for ci = 1:size(configs,1)
    K = configs(ci,1); nt = configs(ci,2); nTest = configs(ci,3);
    if nt + nTest > 64, continue; end

    fwd=zeros(nSplits,1); rev=zeros(nSplits,1); fus=zeros(nSplits,1);
    brk=zeros(nSplits,1); fix=zeros(nSplits,1);

    for m = 1:nSplits
        tc=cell(K,1); sc=cell(K,1);
        for c=1:K
            p=randperm(64);
            tc{c}=p(1:nt); sc{c}=p(nt+1:nt+nTest);
        end
        S=[];Sl=[];T=[];Tl=[];
        for c=1:K
            for i=1:nt; col=(c-1)*64+tc{c}(i); S=[S data(:,col)]; Sl=[Sl c]; end
            for i=1:nTest; col=(c-1)*64+sc{c}(i); T=[T data(:,col)]; Tl=[Tl c]; end
        end

        [~,~,o] = DPGSR_Fusion(S,T,Sl,Tl,nt,1e-1,1e-2,'adaptive',[],[],0.65,0.05);
        [~,pf]=max(o.PF,[],1);[~,pr]=max(o.PR,[],1);
        M=size(T,2); Kk=length(o.classes); Z=o.D;

        % Hard fusion (same as inside DPGSR, but explicit)
        Sh=zeros(Kk,M);
        for j=1:M
            vf=norm(Z(j,Sl==o.classes(pf(j))),2)/(norm(Z(j,:),2)+1e-12);
            PRm=sort(o.PR(:,j),'descend'); PRm=PRm(1)-PRm(2);
            if vf>0.65; alfa=0; elseif PRm>0.05; alfa=1; else alfa=0; end
            Sh(:,j)=(1-alfa)*o.PF(:,j)+alfa*o.PR(:,j);
        end
        [~,ph]=max(Sh,[],1);

        fwd(m)=sum(o.classes(pf)==Tl)/M;
        rev(m)=sum(o.classes(pr)==Tl)/M;
        fus(m)=sum(o.classes(ph)==Tl)/M;
        brk(m)=sum(o.classes(pf)==Tl & o.classes(ph)~=Tl);
        fix(m)=sum(o.classes(pf)~=Tl & o.classes(ph)==Tl);
    end

    fprintf('%-4d %-4d %-4d %8.4f %8.4f %8.4f %8d %8d %8d\n', ...
        K, nt, nTest, mean(fwd), mean(rev), mean(fus), ...
        round(mean(brk)), round(mean(fix)), round(mean(fix)-mean(brk)));
end
