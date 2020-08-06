function [y1] = RSGC(X,c,param)
%  This function implements the method described in the TNNLS journal paper 
%  "Robust Structured Graph Clustering" by Dan Shi, Lei Zhu, Yikun Li,
%  Jingjing Li, and Xiushan Nie.

% Input:
% X is row-based data matrix of size l by n
% c is the number of clusters initialized
% Output: 
% y1 is the predicted label 

[l,n] = size(X);
%% Data processing
X = NormalizeFea(X,0);
% X = mapminmax(X,0,1);
%% Initialization
% initialize U and V
idx_U = kmeans(X',param.r);
U = zeros(n,param.r);
for j = 1:length(idx_U)
    U(j,idx_U(j)) = 1;
end
U = U+0.2; % numerical stability purpose
idx_V = kmeans(X',param.r);
V = zeros(l,param.r);
for i = 1:length(idx_V)
    V(i,idx_V(i)) = 1;
end
V = V+0.2; 

% initialize W and F
eps = 1e-4;
distX = L2_distance_1(X,X);
[distX1, idx] = sort(distX,2);
W = zeros(n);
rr = zeros(n,1); 
for i = 1:n
    di = distX1(i,2:param.k1+2);
    rr(i) = 0.5*(param.k1*di(param.k1+1)-sum(di(1:param.k1)));
    id = idx(i,2:param.k1+2);
    W(i,id) = (di(param.k1+1)-di)/(param.k1*di(param.k1+1)-sum(di(1:param.k1))+eps);
end
if param.alpha <= 0
    param.alpha = mean(rr);
end
param.gamma = mean(rr);

W0 = (W+W')/2;
D0 = diag(sum(W0));
L = D0 - W0;
[F, temp, evs] = eig1(L,c,0);

if sum(evs(1:c+1)) < 0.00000000001
%     error('The original graph has more than %d connected component', c);
    result = [-1,-1,-1];
    return;
end

% initialize lambda1 and lambda2 
lambda1 = zeros(l,n);
lambda2 = zeros(size(U));

pho = 2;
pho2 = 2;
maxiter = 50;
k = 1; % counting number of iterations
flagW = 0;
flagUV = 0;
%% Optimization
while k < maxiter  
    % update E
    temp1 = X-V*U'+1/param.mu*lambda1;
    [E] = L21_solver(temp1,1/param.mu);
    
    % update V
    V = (X-E+1/param.mu*lambda1)*U;                                   
    
    % update Z
    temp2 = U+1/param.mu*lambda2-param.beta/param.mu*(U'*L)';                               
    [Z] = nonneg_L2(temp2);
    
    % update U
    temp3 = Z+1/param.mu*lambda2-param.beta/param.mu*L*Z+(V'*(X-E+1/param.mu*lambda1))';
    [Nu,S,Qu] = mySVD(temp3,0);
    U = Nu*Qu';
    
    % update W
    distf = L2_distance_1(F',F');
    distuz = L2_distance_1(U',U');
    W = zeros(n);
    for i=1:n
        idxa0 = idx(i,2:param.k1+1);
        dfi = distf(i,idxa0);
        dui = distuz(i,idxa0);
        ad = -(dui+param.gamma/param.beta*dfi)/(2*param.alpha);
        W(i,idxa0) = EProjSimplex_new(ad);
    end
    
    % update F
    W = (W+W')/2;
    D = diag(sum(W));
    L = D-W;
    F_old = F;
    [F, temp, ev] = eig1(L, c, 0);
    evs(:,k+1) = ev;

    fn1 = sum(ev(1:c));
    fn2 = sum(ev(1:c+1));
    if fn1 > 0.00000000001
        param.gamma = pho2*param.gamma;
    elseif fn2 < 0.00000000001
        param.gamma = param.gamma/pho2;  
        F = F_old;
    else
        flagW = 1;
    end
    
    % update lambda1 and lambda2
    lambda1 = lambda1+param.mu*(X-V*U'-E);
    lambda2 = lambda2+param.mu*(U-Z);
    param.mu = pho*param.mu;
    
%     bfval(k) = ObjFV(X,U,V,W,F,L,lambda1,lambda2,E,Z,param);    
%     if k>=2
%         aa(k) = abs(bfval(k)-bfval(k-1))/bfval(k-1);
%     end;
%     if k>=2 && abs(bfval(k)-bfval(k-1))/bfval(k-1)<0.0001
%         flagUV = 1;
%         break;
%     end;
     k = k+1;
end
[clusternum, y1]=graphconncomp(sparse(W));
y1 = y1';
% [result] = ClusteringMeasure(y1, y); % clustering results
end