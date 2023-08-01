%% power iteration
function [d, V] = t_power_iter(A)
    [m,l,n] = size(A);

    V = randn(m,1,n);
    V = t_normalize(V);
    k = 10000;
    for i = 1:k
        V = tprod(A,V);
        V = t_normalize(V);
        d = tprod(tran(V),tprod(A,V));
    end
end