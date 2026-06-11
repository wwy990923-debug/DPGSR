% test_subsets2.m — 自训练: 谁强用谁做伪标签
clc; rng(1);
load('pcayaleb50.mat'); data = double(dataset);
K=38; N=64; l1=1e-1; l2=1e-2;

subset_ranges = {1:7, 8:19, 20:31, 32:45, 46:64};
subset_names  = {'Sub1','Sub2','Sub3','Sub4','Sub5'};

fprintf('Self-train: use stronger path for pseudo-labels\n\n');
fprintf('%-6s %8s %8s %8s %8s %8s\n','Train','FwdBase','RevBase','Final','Gain','Teacher');
fprintf('%-6s %8s %8s %8s %8s %8s\n','----','-------','-------','-----','----','-------');

for si=1:5
    tr=subset_ranges{si}; te=setdiff(1:64,tr); nt=length(tr);
    S=[];Sl=[];T=[];Tl=[];
    for c=1:K
        for i=tr; S=[S data(:,(c-1)*N+i)]; Sl=[Sl c]; end
        for i=te; T=[T data(:,(c-1)*N+i)]; Tl=[Tl c]; end
    end
    M=size(T,2);

    % Round 1: check which path is stronger
    [~,~,o]=DPGSR_Fusion(S,T,Sl,Tl,nt,l1,l2,'adaptive',[],[],0.65,0.05);
    [~,pf]=max(o.PF,[],1);[~,pr]=max(o.PR,[],1);
    fwd_base=sum(o.classes(pf)==Tl)/M; rev_base=sum(o.classes(pr)==Tl)/M;
    use_rev = (rev_base > fwd_base);  % 谁强用谁
    teacher = 'Fwd'; if use_rev; teacher='Rev'; end

    unres=true(1,M); pseudo=zeros(1,M);
    for rnd=1:20
        [~,~,o]=DPGSR_Fusion(S,T(:,unres),Sl,Tl(unres),Sl,l1,l2,'adaptive',[],[],0.65,0.05);
        Z=o.D;[~,pf]=max(o.PF,[],1);[~,pr]=max(o.PR,[],1);uri=find(unres);nt_u=sum(unres);
        verif=zeros(1,nt_u);
        for j=1:nt_u; verif(j)=norm(Z(j,Sl==o.classes(pf(j))),2)/(norm(Z(j,:),2)+1e-12); end
        new=(verif>0.65);n_add=sum(new);
        if n_add==0; break; end
        for jj=1:nt_u
            if new(jj)
                j=uri(jj);S=[S,T(:,j)];
                if use_rev; Sl=[Sl,o.classes(pr(jj))]; pseudo(j)=o.classes(pr(jj));
                else;       Sl=[Sl,o.classes(pf(jj))]; pseudo(j)=o.classes(pf(jj));
                end
                unres(j)=false;
            end
        end
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
    fprintf('%-6s %8.4f %8.4f %8.4f %+8.4f %8s\n',subset_names{si},fwd_base,rev_base,final,final-fwd_base,teacher);
end
