% Maximum likelihood estimation.
function [A] = A_matrix(X, N, L, d)
    dim = L+d*L*L;
    A = zeros(dim, dim);
    for i=1:N
        A = A + X*transpose(X);
    end
    A = A.*1/N;