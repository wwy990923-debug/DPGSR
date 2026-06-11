% test_fewshot_self.m — 跨光照few-shot加自训练
clc; rng(1);
load('pcayaleb50.mat'); data = double(dataset);
K=38; N=64; l1=1e-1; l2=1e-2;

subset_ranges = {1:7, 8:19, 20:31, 32:45, 46:64};
subset_names  = {'Sub1','Sub2','Sub3','Sub4','Sub5'};
nt_vals = [2, 4, 6];

fprintf('Cross-lighting Few-shot + Self-training\n');
fprintf('%-6s %3s %8s %8s %8s %6s\n','Train','nt','FwdBase','FusBase','SelfTr','Rnds');
fprintf('%-6s %3s %8s %8s %8s %6s\n','----','--','-------','-------','------','----');

for si=1:5
    tr_src=subset_ranges{si};
    for ni=1:length(nt_vals)
        nt=nt_vals(ni); if nt>length(tr_src); continue; end
        tr=tr_src(1:nt); te=setdiff(1:64,tr);

        S=[];Sl=[];T=[];Tl=[];
        for c=1:K
            for i=tr; S=[S data(:,(c-1)*N+i)]; Sl=[Sl c]; end
            for i=te; T=[T data(:,(c-1)*N+i)]; Tl=[Tl c]; end
        end
        M=size(T,2);
        S0=S; Sl0=Sl;  % save original

        % Single-pass baseline
        [~,~,o]=DPGSR_Fusion(S,T,Sl,Tl,Sl,l1,l2,'adaptive',[],[],0.65,0.05);
        [~,pf]=max(o.PF,[],1); fwd_base=sum(o.classes(pf)==Tl)/M;
        Z=o.D;Sh=zeros(K,M);
        for j=1:M
            vf=norm(Z(j,Sl==o.classes(pf(j))),2)/(norm(Z(j,:),2)+1e-12);
            PRm=sort(o.PR(:,j),'descend');PRm=PRm(1)-PRm(2);
            if vf>0.65;alfa=0;elseif PRm>0.05;alfa=1;else alfa=0;end
            Sh(:,j)=(1-alfa)*o.PF(:,j)+alfa*o.PR(:,j);
        end
        [~,ph]=max(Sh,[],1); fus_base=sum(o.classes(ph)==Tl)/M;

        % Self-training
        S=S0; Sl=Sl0; unres=true(1,M); pseudo=zeros(1,M);
        n_rounds=0;
        for rnd=1:20
            [~,~,o]=DPGSR_Fusion(S,T(:,unres),Sl,Tl(unres),Sl,l1,l2,'adaptive',[],[],0.65,0.05);
            Z=o.D;[~,pf]=max(o.PF,[],1);uri=find(unres);nt_u=sum(unres);
            verif=zeros(1,nt_u);
            for j=1:nt_u; verif(j)=norm(Z(j,Sl==o.classes(pf(j))),2)/(norm(Z(j,:),2)+1e-12); end
            new=(verif>0.65);n_add=sum(new);
            if n_add==0; n_rounds=rnd; break; end
            for jj=1:nt_u;if new(jj);j=uri(jj);S=[S,T(:,j)];Sl=[Sl,o.classes(pf(jj))];unres(j)=false;pseudo(j)=o.classes(pf(jj));end;end
        end

        % Phase 2
        fwd_all=zeros(1,M);
        for jj=1:M; if ~unres(jj); fwd_all(jj)=pseudo(jj); end; end
        for jj=1:nt_u; if unres(uri(jj)); fwd_all(uri(jj))=o.classes(pf(jj)); end; end
        D=Reverse(T,S,Sl,l2); cls_u=unique(Sl,'stable');Kk=length(cls_u);SR=zeros(Kk,M);
        for j=1:M; for kk=1:Kk;idx=(Sl==cls_u(kk));nk=sum(idx);Ak=norm(S(:,idx),'fro');SR(kk,j)=norm(D(j,idx),2)/(sqrt(nk)*Ak+1e-12);end;end
        PR=SR./(sum(SR,1)+1e-12);[~,pr]=max(PR,[],1);
        uri=find(unres);
        for jj=uri
            kF=fwd_all(jj);if kF==0;continue;end;kF_idx=find(cls_u==kF);
            if isempty(kF_idx);pseudo(jj)=cls_u(pr(jj));continue;end
            vf=norm(D(jj,Sl==kF),2)/(norm(D(jj,:),2)+1e-12);
            PRs=sort(PR(:,jj),'descend');PRm=PRs(1)-PRs(2);
            if vf>0.65;pseudo(jj)=kF;elseif PRm>0.05;pseudo(jj)=cls_u(pr(jj));else pseudo(jj)=kF;end
        end
        final=sum(pseudo==Tl)/M;
        fprintf('%-6s %3d %8.4f %8.4f %8.4f %6d\n',subset_names{si},nt,fwd_base,fus_base,final,n_rounds);
    end
end
