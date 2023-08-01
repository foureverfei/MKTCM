%%ss
clear;clc;
it = 1;
%%
dataset_box = {'GLIOMA'};

beta_box = [1e-1];
lambda_box = [0.09 ];
gamma_box = [1e-3];

% dataset_box = {'leukemia'};
% 
% beta_box = [0.09];
% lambda_box = [0.9];
% gamma_box = [0.005];


lambda_box_length = size(lambda_box,2);
beta_box_length = size(beta_box,2);
gamma_box_length = size(gamma_box,2);


for dataset_index = 1:length(dataset_box)
    dataset_name = cell2mat(dataset_box(dataset_index));
    dataset_file = strcat(dataset_name, '.mat');
    load(dataset_file);
    data = X;
    gt = Y;
    for i=1:length(gt)
        if gt(i)==-1
            gt(i) = 2;
        end
    end
    clear X Y;
    %keamn
    num_cluster = length(unique(gt));
    REPlic = 20; % Number of replications for KMeans
    MAXiter = 1000; % Maximum number of iterations for KMeans

    
    %% Construct kernel and transition matrix
    ratio=1;
    sigma(1)=ratio*optSigma(data);   %两两之间距离中值
    cls_num = length(unique(gt));
    num_kernels = 4;
    opts.num_kernels = num_kernels;
    opts.cls_num = cls_num;
    K=[];
    T=cell(1,num_kernels);
    dis_method = {'Gaussian','Polynomial','PolyPlus','Linear'};
    %dis_method = {'Gaussian','Gaussian','Gaussian','Gaussian'};
    for j=1:length(dis_method)
        options.KernelType = dis_method{j};
        options.t = sigma(1);
        K(:,:,j) = constructKernel(data,data,options);
        D=diag(sum(K(:,:,j),2));
        %         L_rw=D^-1*K(:,:,j);
        %         T{j}=L_rw;
        L_rw=D^-0.5*K(:,:,j)*D^-0.5;
        T{j}=L_rw;
    end
    clear L_rw K num_views options D ratio;
    
    
    for j = 1:beta_box_length
        for i = 1:lambda_box_length
            for k = 1:gamma_box_length
                opts.beta = beta_box(j);
                opts.lambda =lambda_box(i);
                opts.gamma = gamma_box(k);
                opts.obj_f = 1;
                method_name(it) = string(strcat(dataset_name,'_MKTCM'));
                Beta(it) = opts.beta;
                Lambda(it) = opts.lambda;
                Gamma(it) = opts.gamma;
                [F(it) P(it) R(it) nmi(it) avgent(it) AR(it) ACC(it) P_hat error] = func_multi_kernel_manifold(T,gt,dataset_name,opts);
                it = it +1;
            end
        end
    end
    finale_result = [method_name',Beta',Lambda',Gamma',ACC',nmi',AR',F',P',R',avgent'];
    disp(finale_result);
    save_path = '...\Results\';  %% ur save path 
    xlswrite(string(save_path)+'MKTC_manifold_'+dataset_name+'_result.xls', finale_result);
    save_path_name = strcat('...\Results\',method_name(it-1)); %% ur save path 
    save(save_path_name);
end



