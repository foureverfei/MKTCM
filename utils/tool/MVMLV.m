function [S,w] = MVMLV(Phat,opts)

%Input:
% X    -- cell, multi-view data points
% X{i} -- i-th view matrix, row for feature, column for sample d*n
% opts -- parameter settings:beta gamma 

%Output:
% S    --S Affinity Matrix n*n
% w   --the weights of each view

num_omics = length(Phat);
n = size(Phat{1},2);
%%
% setting default parameters
beta = 1;

%initial S S1 S2,and w;
S=zeros(n,n);
S1=S;
S2=S;
w=1/num_omics*ones(num_omics,1);

%initial parameters related to ADMM
iter = 0;
num_iter = 200;
mu1=1e1;
mu2=1e1;
Lambda1 = zeros(n,n);
Lambda2 = zeros(n,n);
err_thr = 1e-4;
rho = 1.2;
max_mu = 1e6;

if isfield(opts, 'beta');         beta = opts.beta;	end

objw=zeros(1,num_iter);
objzuz=zeros(1,num_iter);
obj=zeros(1,num_iter);
%%
while(iter < num_iter)
    A=zeros(n,n);
    for v=1:num_omics
    A=w(v)*Phat{v}+A;
    end
    iter=iter+1;
    % Update matrix S
    S=(mu1*(S1-Lambda1/mu1)+mu2*(S2-Lambda2/mu2)+A)/(mu1+mu2);
    % Update matrix S1
    S1=zeros(n,n);
    for i=1:n
        S1(:,i)=EProjSimplex_new(S(:,i)+Lambda1(:,i)/mu1);
    end
    % Update matrix S2
    S2tmp=S+Lambda2/mu2;
    S2tmpDiag=diag(diag(S2tmp));
    S2tmp=S2tmp-S2tmpDiag;
    S2=max((S2tmp+S2tmp')/2,0);
    % Update vector w
    zsz=zeros(num_omics,1);
    for v=1:num_omics
        zsz(v)=trace(Phat{v}*S);
    end
    w=Hbeta(zsz,1/beta);   
    % Update parameters mu1, mu2, Lambda1, and Lambda2
    mu1=min(rho*mu1,max_mu);
    mu2=min(rho*mu2,max_mu);
    Lambda1=Lambda1+mu1*(S-S1);
    Lambda2=Lambda2+mu2*(S-S2);
    % Convergence condition reached 
    for v=1:num_omics
        objw(iter)=beta*w(v)*log(w(v))+objw(iter);
    end
    objzuz(iter)=trace(A*S);
    obj(iter)=objw(iter)+objzuz(iter);
    if(iter>=2 && max([norm(S-S1,'inf'),norm(S-S2,'inf')])<=err_thr)
        break
    end
end
end
%%
function P = Hbeta(D, beta)
D = (D-min(D))/(max(D) - min(D)+eps);
P = exp(D * beta);
sumP = sum(P);
P = P / sumP;
end