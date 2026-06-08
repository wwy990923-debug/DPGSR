% test_verif_soft.m — 验证分硬开关 vs 软权重
clc; rng(1);
load('pcayaleb50.mat'); data = double(dataset);
K=15; N=64; nT=8; nS=5;
thr_vals = [0.5, 0.55, 0.6, 0.65, 0.7];

fprintf('Soft & Hard verification gate (PCA-50, l1=1e-1, l2=1e-2)\n\n');
fprintf('%-8s %8s %8s %8s\n', 'th', 'FwdAcc', 'HardFus', 'SoftFus');
fprintf('%-8s %8s %8s %8s\n', '--', '------', '-------', '-------');

for ti = 1:length(thr_vals)
    th = thr_vals(ti);
    fwd=zeros(nS,1); fus_h=zeros(nS,1); fus_s=zeros(nS,1);

    for m = 1:nS
        tc=cell(K,1); sc=cell(K,1);
        for c=1:K; p=randperm(N); tc{c}=p(1:nT); sc{c}=p(nT+1:end); end
        S=[]; Sl=[]; T=[]; Tl=[];
        for c=1:K
            for i=1:nT; col=(c-1)*N+tc{c}(i); S=[S data(:,col)]; Sl=[Sl c]; end
            for i=1:(N-nT); col=(c-1)*N+sc{c}(i); T=[T data(:,col)]; Tl=[Tl c]; end
        end

        [~,~,o] = DPGSR_Fusion(S,T,Sl,Tl,nT,1e-1,1e-2,'adaptive',[],[],th,0.05);
        [~,pf] = max(o.PF,[],1); M=size(T,2);
        fwd(m) = sum(o.classes(pf)==Tl)/M;
        Z = o.D; Kk = length(o.classes);

        % Hard fusion (alpha = 0 or 1)
        Score_h = zeros(Kk,M);
        for j=1:M
            verif = norm(Z(j,Sl==o.classes(pf(j))),2)/(norm(Z(j,:),2)+1e-12);
            PRm = sort(o.PR(:,j),'descend'); PRm = PRm(1)-PRm(2);
            if verif > th; alfa=0;
            elseif PRm > 0.05; alfa=1;
            else alfa=0; end
            Score_h(:,j) = (1-alfa)*o.PF(:,j) + alfa*o.PR(:,j);
        end
        [~,ph] = max(Score_h,[],1);
        fus_h(m) = sum(o.classes(ph)==Tl)/M;

        % Soft fusion (alpha = 1 - verif, continuous)
        Score_s = zeros(Kk,M);
        for j=1:M
            verif = norm(Z(j,Sl==o.classes(pf(j))),2)/(norm(Z(j,:),2)+1e-12);
            alfa = 1 - verif;
            Score_s(:,j) = (1-alfa)*o.PF(:,j) + alfa*o.PR(:,j);
        end
        [~,ps] = max(Score_s,[],1);
        fus_s(m) = sum(o.classes(ps)==Tl)/M;
    end

    fprintf('%-8.2f %8.4f %8.4f %8.4f\n', th, mean(fwd), mean(fus_h), mean(fus_s));
end
