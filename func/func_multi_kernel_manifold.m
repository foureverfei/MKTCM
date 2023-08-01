function [F P R nmi avgent AR ACC P_hat error z_tensor_final] = func_multi_kernel_manifold_v2(K,gt,data_name,opts)
%FUNC_multi_kernel_ 此处显示有关此函数的摘要
%   此处显示详细说明
    error = [];
    lambda = 0.04;
    beta = 0.04;
    obj_f = 0;
    if isfield(opts, 'lambda');         lambda = opts.lambda;     end
    if isfield(opts, 'beta');           beta = opts.beta;     end
    if isfield(opts, 'gamma');           gamma = opts.gamma;     end
    if isfield(opts, 'num_kernels');           num_kernels = opts.num_kernels;     end
    if isfield(opts, 'cls_num');           cls_num = opts.cls_num;     end
    if isfield(opts, 'obj_f');             obj_f = opts.obj_f;    end
    K_tensor = cat(3, K{:,:});
    data_num = size(K_tensor,1);
    k = K_tensor(:);

    %% initial
    N = size(K{1},1); % number of samples
    for i=1:num_kernels
        T{i} = zeros(N,N);   
        E{i} = zeros(N,N);
        B{i} = zeros(N,N);
        Y_1{i} = zeros(N,N);
        Y_2{i} = zeros(N,N);
    end
    W = zeros(N,N);
    Z = zeros(N,N);
    Y_3 = zeros(N,N);
    
    T_tensor = cat(3, T{:,:}); 
    E_tensor = cat(3, E{:,:});
    
    sX = [N, N, num_kernels];
    alpha = ones(num_kernels,1)/num_kernels;
    tol = 1e-6;
    iter = 1;
    mu1 = 10e-3;
    mu2 = 10e-3;
    mu3 = 10e-3;
    max_mu = 10e10;
    pho_mu = 2;
    max_iter=200;
    
    while iter < max_iter
        Tpre=T_tensor;
        Epre=E_tensor;
        Wpre = W;
        fprintf('----processing iter %d--------\n', iter+1);
        %% update T
        Y_1_tensor = cat(3, Y_1{:,:});
        y_1 = Y_1_tensor(:);
        e = E_tensor(:);
        
        [t, objV] = wshrinkObj(k - e + 1/mu1*y_1,1/mu1,sX,0,3)   ;
        T_tensor = reshape(t, sX);
        T{1}=T_tensor(:,:,1);
        T{2}=T_tensor(:,:,2);
        T{3}=T_tensor(:,:,3);
        T{4}=T_tensor(:,:,4);
        %% update E
        F = [K{1}-T{1}+Y_1{1}/mu1;K{2}-T{2}+Y_1{2}/mu1;K{3}-T{3}+Y_1{3}/mu1;K{4}-T{4}+Y_1{4}/mu1];
        [Econcat] = solve_l1l2(F,beta/mu1);
        
        E{1} = Econcat(1:size(K{1},1),:);
        E{2} = Econcat(size(K{1},1)+1:size(K{1},1)+size(K{2},1),:);
        E{3} = Econcat(size(K{1},1)+size(K{2},1)+1:size(K{1},1)+size(K{2},1)+size(K{3},1),:);
        E{4} = Econcat(size(K{1},1)+size(K{2},1)+size(K{3},1)+1:end,:);
        E_tensor = cat(3, E{:,:});
        
        for i=1:num_kernels
            Y_1{i} = Y_1{i} + mu1*(K{i}-T{i}-E{i});
        end
        %% Update B
        for i = 1:num_kernels
            D = T{i}-1/mu2*Y_2{i};
            M = D+lambda*alpha(i)/mu2*(W'+W);
            [U, S, V] = svd(M,'econ');
            S = max(S,0);
            B{i} = U*S*V';
        end
        clear D M U S V;
        %% Update W
        sum_avB = zeros(N,N);
        for i = 1:num_kernels
            sum_avB = sum_avB+ alpha(i)*(B{i}+B{i}');
        end
        sum_avB = lambda * sum_avB;
        W_temp = 1/mu3*(sum_avB+mu3*Z-Y_3);
        W = project_fantope(W_temp,num_kernels);
        clear sum_avB W_temp;
        %% Update Z
        D = W+1/mu3*Y_3;
        threshold = gamma/mu3;
        D_b_index = D>threshold;
        D_b = D - threshold;
        D_b = D_b .* D_b_index;
        D_s_index = D<-threshold;
        D_s = D + threshold;
        D_s = D_s .* D_s_index;
        Z = D_b + D_s;
        
        clear D threshold;
        %% Update alpha
        sum_alpha = sum(alpha);
        alpha = alpha/sum_alpha;
        clear sum_alpha;
        %% Update Y1 Y2 Y3
        for i = 1:num_kernels
            Y_1{i} = Y_1{i} + mu1*(K{i}-T{i}-E{i});
            Y_2{i} = Y_2{i} + mu2*(B{i}-T{i});
        end
        Y_3 = Y_3 + mu3*(W-Z);
        %% check convergence
        leq = K_tensor-T_tensor-E_tensor;
        leqm = max(abs(leq(:)));
        difT = max(abs(T_tensor(:)-Tpre(:)));
        difE = max(abs(E_tensor(:)-Epre(:)));
        err = max([leqm,difT,difE]);
        fprintf('iter = %d, mu1 = %.3f, difZ = %.3f, difE = %.8f,err=%d\n'...
            , iter,mu1,difT,difE,err);
        if err < tol
            break;
        end
        %error(iter) = err;
        if obj_f ==1
            if iter >1
                error_1(iter-1) = 0;error_2(iter-1)=0;error_3(iter-1)=0;
                for i = 1:num_kernels
                    error_1(iter-1) = error_1(iter-1) + norm(T{i}-Tpre(:,:,i),'fro')^2;
                    error_2(iter-1) = error_2(iter-1) + norm(E{i}-Epre(:,:,i),'fro')^2;
                    error_3(iter-1) = error_3(iter-1) + norm(K{i}-T{i}-E{i},'fro')^2;
                end
                error_4(iter-1) = norm(W-Wpre,'fro')^2;
                error(iter-1) = max([error_1(iter-1) error_2(iter-1) error_3(iter-1) error_4(iter-1)]);
            end
        else
            error = 0;
        end
        iter = iter + 1;
        mu1 = min(mu1*pho_mu, max_mu);
        mu2 = min(mu2*pho_mu, max_mu);
        mu3 = min(mu3*pho_mu, max_mu);
    end
    P_hat = max(real(W),0);
    fprintf(data_name)

    [pred] = SpectralClustering(P_hat, cls_num);
    [F P R nmi avgent AR ACC] = baseline_acc(pred,gt);
    fprintf('F=%f, P=%f, R=%f, nmi score=%f, avgent=%f,  AR=%f, ACC=%f,\n',F(1),P(1),R(1),nmi(1),avgent(1),AR(1),ACC(1));
    end

