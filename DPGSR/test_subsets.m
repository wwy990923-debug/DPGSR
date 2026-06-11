% test_subsets.m — Yale B 光照子集评测 + 自训练
clc; rng(1);
load('pcayaleb50.mat'); data = double(dataset);
K=38; N=64; l1=1e-1; l2=1e-2;

subset_ranges = {1:7, 8:19, 20:31, 32:45, 46:64};
subset_names  = {'Sub1(0-10deg)','Sub2(10-25)','Sub3(25-50)','Sub4(50-77)','Sub5(>77)'};

% ==== Part A: Single-pass ====
fprintf('===== Single-Pass DP-GSR =====\n');
fprintf('%-16s %4s %5s %8s %8s %8s %8s\n','Train','nt','nTst','Fwd','Rev','Fus','Gain');
fprintf('%-16s %4s %5s %8s %8s %8s %8s\n','----','--','----','---','---','---','----');
for si=1:5
    tr=subset_ranges{si}; te=setdiff(1:64,tr); nt=length(tr);
    S=[];Sl=[];T=[];Tl=[];
    for c=1:K
        for i=tr; S=[S data(:,(c-1)*N+i)]; Sl=[Sl c]; end
        for i=te; T=[T data(:,(c-1)*N+i)]; Tl=[Tl c]; end
    end
    M=size(T,2);
    [~,~,o]=DPGSR_Fusion(S,T,Sl,Tl,nt,l1,l2,'adaptive',[],[],0.65,0.05);
    [~,pf]=max(o.PF,[],1);[~,pr]=max(o.PR,[],1);
    fwd=sum(o.classes(pf)==Tl)/M; rev=sum(o.classes(pr)==Tl)/M;
    Z=o.D; Sh=zeros(K,M);
    for j=1:M
        vf=norm(Z(j,Sl==o.classes(pf(j))),2)/(norm(Z(j,:),2)+1e-12);
        PRm=sort(o.PR(:,j),'descend');PRm=PRm(1)-PRm(2);
        if vf>0.65;alfa=0;elseif PRm>0.05;alfa=1;else alfa=0;end
        Sh(:,j)=(1-alfa)*o.PF(:,j)+alfa*o.PR(:,j);
    end
    [~,ph]=max(Sh,[],1); fus=sum(o.classes(ph)==Tl)/M;
    fprintf('%-16s %4d %5d %8.4f %8.4f %8.4f %+8.4f\n',subset_names{si},nt,M,fwd,rev,fus,fus-fwd);
end

% ==== Part B: Self-training on all subsets ====
fprintf('\n===== Self-Training Pipeline =====\n');
fprintf('%-16s %6s %8s %8s %8s\n','Train','Rounds','FwdBase','Final','Gain');
fprintf('%-16s %6s %8s %8s %8s\n','----','------','-------','-----','----');
for si=1:5
    tr=subset_ranges{si}; te=setdiff(1:64,tr); nt=length(tr);
    S=[];Sl=[];T=[];Tl=[];
    for c=1:K
        for i=tr; S=[S data(:,(c-1)*N+i)]; Sl=[Sl c]; end
        for i=te; T=[T data(:,(c-1)*N+i)]; Tl=[Tl c]; end
    end
    M=size(T,2); unres=true(1,M); pseudo=zeros(1,M);
    fwd_base=0; n_rounds=0;

    for rnd=1:20
        [~,~,o]=DPGSR_Fusion(S,T(:,unres),Sl,Tl(unres),Sl,l1,l2,'adaptive',[],[],0.65,0.05);
        Z=o.D;[~,pf]=max(o.PF,[],1);uri=find(unres);nt_u=sum(unres);
        if rnd==1; fwd_base=sum(o.classes(pf)==Tl(uri))/nt_u; end
        verif=zeros(1,nt_u);
        for j=1:nt_u; verif(j)=norm(Z(j,Sl==o.classes(pf(j))),2)/(norm(Z(j,:),2)+1e-12); end
        new=(verif>0.65);n_add=sum(new);
        if n_add==0; n_rounds=rnd; break; end
        for jj=1:nt_u;if new(jj);j=uri(jj);S=[S,T(:,j)];Sl=[Sl,o.classes(pf(jj))];unres(j)=false;pseudo(j)=o.classes(pf(jj));end;end
    end

    % Collect forward predictions for ALL samples
    fwd_all=zeros(1,M);
    for jj=1:M
        if ~unres(jj); fwd_all(jj)=pseudo(jj); end
    end
    for jj=1:nt_u
        if unres(uri(jj)); fwd_all(uri(jj))=o.classes(pf(jj)); end
    end

    % Phase 2: Reverse + fusion
    D=Reverse(T,S,Sl,l2);
    cls_u=unique(Sl,'stable');Kk=length(cls_u);SR=zeros(Kk,M);
    for j=1:M
        for kk=1:Kk;idx=(Sl==cls_u(kk));nk=sum(idx);Ak=norm(S(:,idx),'fro');SR(kk,j)=norm(D(j,idx),2)/(sqrt(nk)*Ak+1e-12);end
    end
    PR=SR./(sum(SR,1)+1e-12);[~,pr]=max(PR,[],1);
    uri=find(unres);
    for jj=uri
        kF=fwd_all(jj);if kF==0;continue;end
        kF_idx=find(cls_u==kF);
        if isempty(kF_idx);pseudo(jj)=cls_u(pr(jj));continue;end
        vf=norm(D(jj,Sl==kF),2)/(norm(D(jj,:),2)+1e-12);
        PRs=sort(PR(:,jj),'descend');PRm=PRs(1)-PRs(2);
        if vf>0.65;pseudo(jj)=kF;elseif PRm>0.05;pseudo(jj)=cls_u(pr(jj));else pseudo(jj)=kF;end
    end
    final=sum(pseudo==Tl)/M;
    fprintf('%-16s %6d %8.4f %8.4f %+8.4f\n',subset_names{si},n_rounds,fwd_base,final,final-fwd_base);
end
