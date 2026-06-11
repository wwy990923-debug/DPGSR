% test_pipeline2.m — 自训练后只跑逆向+融合
clc; rng(1);
load('pcayaleb50.mat'); data = double(dataset);
K=38; N=64; nT=8; l1=1e-1; l2=1e-2;

tc=cell(K,1); sc=cell(K,1);
for c=1:K; p=randperm(N); tc{c}=p(1:nT); sc{c}=p(nT+1:end); end
S=[];Sl=[];T=[];Tl=[];
for c=1:K
    for i=1:nT; col=(c-1)*N+tc{c}(i); S=[S data(:,col)]; Sl=[Sl c]; end
    for i=1:(N-nT); col=(c-1)*N+sc{c}(i); T=[T data(:,col)]; Tl=[Tl c]; end
end
M=size(T,2); unres=true(1,M); pseudo=zeros(1,M);

% ---- Phase 1: Forward self-training (same as before) ----
fprintf('=== Phase 1: Forward self-training ===\n');
for rnd=1:20
    [~,~,o]=DPGSR_Fusion(S,T(:,unres),Sl,Tl(unres),Sl,l1,l2,'adaptive');
    [~,pf]=max(o.PF,[],1);uri=find(unres);nt=sum(unres);
    verif=zeros(1,nt);
    for j=1:nt; verif(j)=norm(o.D(j,Sl==o.classes(pf(j))),2)/(norm(o.D(j,:),2)+1e-12); end
    new=(verif>0.65);n_add=sum(new);
    fprintf('  R%d: added %d',rnd,n_add);
    if n_add>0; fprintf(' (%.0f%% ok)\n',100*sum(o.classes(pf(new))==Tl(uri(new)))/n_add);
    else; fprintf(' -> converged\n'); break; end
    for jj=1:nt;if new(jj);j=uri(jj);S=[S,T(:,j)];Sl=[Sl,o.classes(pf(jj))];unres(j)=false;pseudo(j)=o.classes(pf(jj));end;end
end
fwd_last=pf;  % forward predictions on last round's unresolved

% ---- Phase 2: Only reverse + fusion (skip forward!) ----
fprintf('\n=== Phase 2: Reverse-only + fusion ===\n');
D = Reverse(T, S, Sl, l2);  % ALL 2128 test as dict → expanded S
M_test=size(T,2); N_train=size(S,2);

% Compute PR from D (same as inside DPGSR_Fusion)
classes=unique(Sl,'stable'); Kk=length(classes);
SR=zeros(Kk,M_test);
for j=1:M_test
    for kk=1:Kk
        idx=(Sl==classes(kk)); nk=sum(idx);
        Ak=norm(S(:,idx),'fro');
        ak=norm(D(j,idx),2);
        SR(kk,j)=ak/(sqrt(nk)*Ak+1e-12);
    end
end
PR=SR./(sum(SR,1)+1e-12);
[~,pr]=max(PR,[],1);

% Get forward predictions for ALL samples
% Confident: pseudo-label; Hard: fwd_last from Phase 1
fwd_all=zeros(1,M_test);
uri=find(~unres);  % already resolved
for jj=1:length(uri); fwd_all(uri(jj))=pseudo(uri(jj)); end
uri_hard=find(unres);
for jj=1:length(uri_hard); fwd_all(uri_hard(jj))=classes(fwd_last(jj)); end

% Fusion on hard samples only
uri=find(unres); nt=sum(unres);
n_fwd=0;n_rev=0;fwd_ok=0;rev_ok=0;
for jj=uri
    kF=fwd_all(jj);  % forward prediction (class label)
    kF_idx=find(classes==kF);
    idx_F=(Sl==kF);
    vf=norm(D(jj,idx_F),2)/(norm(D(jj,:),2)+1e-12);
    PRs=sort(PR(:,jj),'descend');PRm=PRs(1)-PRs(2);
    if vf>0.65; alfa=0; elseif PRm>0.05; alfa=1; else alfa=0; end
    if alfa==0
        pseudo(jj)=kF;n_fwd=n_fwd+1;
        if kF==Tl(jj);fwd_ok=fwd_ok+1;end
    else
        pseudo(jj)=classes(pr(jj));n_rev=n_rev+1;
        if classes(pr(jj))==Tl(jj);rev_ok=rev_ok+1;end
    end
end
fprintf('  pureFWD %d (acc=%.3f), pureREV %d (acc=%.3f)\n',n_fwd,fwd_ok/max(1,n_fwd),n_rev,rev_ok/max(1,n_rev));

% ---- Final ----
final=sum(pseudo==Tl)/M;
fprintf('\nFinal: %.4f | Baseline: 0.7430 | Gain: %+.4f\n',final,final-0.7430);
