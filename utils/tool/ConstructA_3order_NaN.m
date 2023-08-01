function Phat=ConstructA_3order_NaN(X,gamma,clusternum)
% Construct high order similarity matrix and integrate clustering information
%Input:
% X -- the input multi-omics dataset
% gamma -- the number of neighbors in an intermediary station
% clusternum -- the number of clusters

%Output:
% Phat -- a high order similarity matrix

num_views = length(X);
n = size(X{1},2);
% Generate the first order proximity
for v=1:num_views
     options.t=optSigma(X{v}');
     options.KernelType='Gaussian';
     W{v} = constructKernel(X{v}',[],options);
     Z{v}=FindDominateSet(W{v});
     W{v}=(Z{v}+Z{v}')/2;
     clusterinfo{v}=SpectralClustering(W{v},clusternum);
end
% clustermatrix is the overall cluster information Omega
clustermatrix=repmat({zeros(n,n)},1,num_views);
parfor v=1:num_views
    for i=1:n
        for j=find(clusterinfo{v}==clusterinfo{v}(i))
            clustermatrix{v}(i,j)=1;
        end
    end
end
% Generate the intermediary station
for v=1:num_views
 distance=dist2(Z{v}, Z{v});
 [sorted,index] = sort(distance);
 neighborhood{v}= index(2:(gamma+1),:);
end
% Integrate the cluster information 
for v=1:num_views
          W{v}=W{v}.*clustermatrix{v};
end
% Calculate the high order similarity matrix with cluster information
Phat=repmat({zeros(n,n)},1,num_views);
parfor v=1:num_views
    for i=1:n    
       Phat{v}=Phat{v}+repmat(W{v}(:,i),1,gamma)*W{v}(:,neighborhood{v}(:,i))'+W{v}(:,neighborhood{v}(:,i))*repmat(W{v}(:,i),1,gamma)';
    end
end
end

%% Adopt a natural neighbor search approach to find dominate set
function W = FindDominateSet(W)
[m,n]=size(W);
W=NaN_Search(W);
W=W./repmat(sum(W,2)+eps,1,n);
end

function W_final_v1=NaN_Search(W)
[N,~]=size(W);
[sdist,index]=sort(W,2,'descend');%Sort the distances
%NaN-Searching algorithm
r=1;
flag=0;         
nb=zeros(1,N);  %The number of each point's reverse neighbor
count=0;        
count1=0;    
while flag==0
    for i=1:N
        k=index(i,r+1);
        nb(k)=nb(k)+1;
       RNN(k,nb(k))=i;
    end
    r=r+1;
    count2=0;
    for i=1:N
        if nb(i)==0
            count2=count2+1;
        end
    end
    if count1==count2
        count=count+1;
    else
        count=1;
    end
    if count2==0 %|| (r>2 && count>=2)   %The terminal condition
        flag=1;
    end
    count1=count2;
end
I=find(nb<5);
RNN(I,1:5)=index(I,1:5);
nb(nb<5)=5;
W_final_v1=zeros(N,N);
for i=1:N
    W_final_v1(i,RNN(i,1:nb(i)))=W(i,RNN(i,1:nb(i)));
    W_final_v1(i,i)=W(i,i);
end
end
