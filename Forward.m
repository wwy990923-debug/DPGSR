function [W] = Forward(S,T,groupInfo,lambda)
% This routine solves the following nuclear-norm optimization problem,
% which is more general than "lrr.m"
% min |Z|_*+lambda*|E|_2,1
% s.t., X = AZ+E
% inputs:
%        X -- D*N data matrix, D is the data dimension, and N is the number
%             of data vectors.
%        A -- D*M matrix of a dictionary, M is the size of the dictionary
tol = 1e-4;
maxIter = 500;
[d, n] = size(S);
[d, p] = size(T);

if numel(groupInfo) == 1
    nt = groupInfo;
    if mod(n, nt) ~= 0
        error('The number of columns in S must be divisible by nt.');
    end
    S_label = repelem(1:(n/nt), nt);
else
    S_label = groupInfo(:)';
end

if length(S_label) ~= n
    error('Length of S_label/groupInfo must equal the number of columns in S.');
end
classes = unique(S_label, 'stable');
% m = size(appdesigner,2);
rho = 1.2;
max_mu = 1e20;
mu = 1e-7;
mu = 1.1;





%% Initializing optimization variables
% intialize
% for i = 1:2



W = zeros(n,p);
F = zeros(n,p);

Y1 = zeros(n,p);

%% Start main loop
iter = 0;
obj_values = zeros(maxIter, 1); % 存储目标函数值
loss_values = zeros(maxIter, 1); % 存储损失值
while iter<maxIter
    iter = iter + 1;

    %update F
    temp = W - Y1/mu;
   

    F = zeros(n, p);
    for kk = 1:length(classes)
        idx = (S_label == classes(kk));
        F(idx, :) = solve_L21(temp(idx, :), lambda/mu);
    end



    
    
    %udpate W
    
    inv_b = inv(2*S'*S+mu*eye(n));
    W =inv_b*(2*S'*T+mu*(F+Y1/mu));
    
   

    leq1 = F-W;
     stopC = max(max(max(abs(leq1))),max(max(abs(leq1))));

    loss_values(iter) = stopC;
     % compute objective value
    obj_val = norm(S*W - T, 'fro')^2;
    for kk = 1:length(classes)
        idx = (S_label == classes(kk));
        obj_val = obj_val + lambda * sum(sqrt(sum(W(idx, :).^2, 1)));
    end
    obj_values(iter) = obj_val; 
   
    
%     if stopC>stop
%         break;
%     else
%         stop = stopC;
%     end
    if (iter==1 || mod(iter,1)==0 || stopC<tol)
        disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ...
            ',rank=' num2str(rank(W,1e-4*norm(W,2))) ',stopALM=' num2str(stopC,'%2.3e') ]);
    end
    if stopC<tol 
        break;
    else
        
        Y1 = Y1 + mu*leq1;
  
        
        mu = min(max_mu,mu*rho);
    end
end
% % 只保留前 `iter` 次的损失值
% loss_values = loss_values(1:iter);
% 
% % 绘制损失收敛曲线
% norm_obj = loss_values / loss_values(1);
% figure;
% plot(1:iter, norm_obj, '-o', 'LineWidth', 1.5);
% xlabel('Number of Iterations');
% ylabel('Termination Values');
% grid on;

% saveas(gcf, 'iteration4.eps', 'epsc');

norm_obj = obj_values / obj_values(1);
% figure;
% plot(1:iter, norm_obj(1:iter), 'LineWidth', 1.5);
% xlabel('Number of Iterations');
% ylabel('Objective Function Value');
% grid on;
% saveas(gcf, 'objective_curve.eps', 'epsc');


end



function [E] = solve_L21(M,lambda)
n = size(M,2);
E = M;
for i=1:n
    E(:,i) = solve_l2(M(:,i),lambda);
end
end

function [x] = solve_l2(w,lambda)
% min lambda |x|_2 + |x-w|_2^2
nw = norm(w);
if nw>lambda
    x = (nw-lambda)*w/nw;
else
    x = zeros(length(w),1);
end

end



function [E] = solve_L12(M,lambda)
n = size(M,1);
E = M;
for i=1:n
    E(i,:) = solve_l1(M(i,:),lambda);
end
end

function [x] = solve_l1(w,lambda)
% min lambda |x|_2 + |x-w|_2^2
nw = norm(w);
if nw>lambda
    x = (nw-lambda)*w/nw;
else
    x = zeros(1,length(w));
end

end



function [E] = solve_Q(M,lambda,nt)
n = size(M,2);

E = M;
for i=1:n
    E(:,i) = solve_l21(M(:,i),lambda,nt);
end
end

function [x] = solve_l21(w,lambda,nt)
% min lambda |x|_2 + |x-w|_2^2
m = size(w,1);
nw = 0;
for k = 1:m/nt
    nw = nw + norm(w((k-1)*nt+1:k*nt),2);
end
nw = nw^2;
if nw>lambda
    x = (nw-lambda)*w/nw;
else
    x = zeros(length(w),1);
end

end


function [E] = solve_212(M,lambda,nt)
n = size(M,2);

E = M;
for i=1:n
    E(:,i) = solve_mix(M(:,i),lambda,nt);
end
end

function [x] = solve_mix(w,lambda,nt)
% min lambda |x|_2 + |x-w|_2^2
m = size(w,1);
nw = 0;
nwc = [];
x = [];
for k = 1:m/nt
    nwc = [nwc norm(w((k-1)*nt+1:k*nt),2)];
end
nw = norm(nwc,1);
for k = 1:m/nt
    if norm(w((k-1)*nt+1:k*nt))>0
        yk = w((k-1)*nt+1:k*nt);
        xk = yk-2*lambda*nw*yk/(m/nt*lambda*2+1)/nwc(1,k);
        x = [x;xk];
    else
        x = [x;zeros(nt,1)];
    end
end


end


