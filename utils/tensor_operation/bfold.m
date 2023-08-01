function T_hat = bfold(T)
%BFOLD transform matrix T to 3-order tensor 

    [m,n] = size(T);
    T_hat = zeros(n,n,n);
    if n^2 == m
        for i = 1:n
            T_hat(:,:,i) = T((i-1)*n+1:i*n,:);
        end
    else
        disp('row col_num not match')
    end
end

