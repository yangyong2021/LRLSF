%{
Multi-label feature selection based on stable label relevance and label-specific features
input:
X:feature matrix with n rows and d columns
Y:label matrix with n rows and c columns
optmParameter:
    lambda1,lambda2,lambda3,lambda4,gamma:balance parameters
    maxIter:maximum iteration number
    minimumLossMargin:

output:loss threshold
W:projection matrix with d rows and c columns
obj:values of the objective function during the iteration
time2:The running time of LRLSF
%}
function [W,loss,time2] = LRLSF( X, Y, optmParameter)
t = clock;
warning off;
%% optimization parameters
lambda1           = optmParameter.lambda1;
lambda2           = optmParameter.lambda2;
lambda3           = optmParameter.lambda3;
lambda4           = optmParameter.lambda4;
gamma             = optmParameter.gamma;

maxIter           = optmParameter.maxIter;
miniLossMargin    = optmParameter.minimumLossMargin;
%% initializtion
[num_data,num_dim] = size(X);
num_labels=size(Y,2);
XTX = X'*X;
XTY = X'*Y;

W   = (XTX + gamma*eye(num_dim)) \ (XTY);
W_1 = W;

% S = ones(num_labels);
S = rand(num_labels,num_labels);
S_1 = S;

A = 10000 * eye(num_data);
% A = eye(num_data);

options = [];
options.Metric = 'Euclidean';
options.NeighborMode = 'KNN';
options.k = 10;  % nearest neighbor
options.WeightMode = 'HeatKernel';
options.t = 1;
C = constructW(X,options);
L = diag(sum(C,2))-C;
L = full(L);
iter    = 1;
oldloss = 0;



bk   = 1;
bk_1 = 1;

Lip = norm(XTX)^2 + 3*norm(lambda1*Y'*Y)^2  + 3*norm(lambda1*(Y'*(L'+L)*Y))^2 + 3*norm(lambda2*(Y'*A*Y))^2;
Lip = sqrt(Lip);
%% proximal gradient
while iter <= maxIter
% %     S    = (S+S')/2;
    %% update W
    W_k    = W + (bk_1 - 1)/bk * (W - W_1);
    Gw_s_k = W_k - 1/Lip * (XTX*W_k  - X'*Y*S);
    W_1    = W;
    W      = softthres(Gw_s_k,lambda3/Lip);
    W = max(W,0);
    %% update S
    S_k  = S + (bk_1 - 1)/bk * (S - S_1);
    Gs_k = S_k - 1/Lip * (lambda1*(Y'*Y*S_k-Y'*X*W) + lambda1*(Y'*L*Y*S_k) + lambda2*(Y'*A*Y*S_k - Y'*A*Y));
    S_1  = S;
    S    = softthres(Gs_k,lambda4/Lip);
    S    = max(S,0);
%     S    = (S+S')/2;
    
    bk_1   = bk;
    bk     = (1 + sqrt(4*bk^2 + 1))/2;
    
    M=X*W-Y*S;
    predictionLoss   = 0.5*trace(M'*M);
    YS=Y*S;
    
    GlobalLC         = 0.5*trace((YS-Y)'*A*(YS-Y));
    
    LocalLC          = 0.5*trace(YS'*L*YS);
    
    sparsityW        = sum(sum(W~=0));
    sparsityS        = sum(sum(S~=0));
    
    totalloss        = predictionLoss + lambda2*GlobalLC + lambda1*LocalLC+ lambda3*sparsityW + lambda4*sparsityS;
    loss(iter)     = totalloss;
%     disp("求解W--obj---当前迭代次数："+string(iter));
%     disp("obj："+string(loss(iter)))
    if abs((oldloss - totalloss)/oldloss) <= miniLossMargin
        break;
    elseif totalloss <=0
        break;
    else
        oldloss = totalloss;
    end
    iter=iter+1;
end
 plot(loss,'*-r');
 time2 = etime(clock,t);
end

%% soft thresholding operator
function W = softthres(W_t,lambda)
W = max(W_t-lambda,0) - max(-W_t-lambda,0);
end
