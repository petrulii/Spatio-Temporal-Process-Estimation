function [approx, x] = lse_lin_approx(r, eps, x_init)
    approx = zeros(r+1,2);
    x = zeros(1,r);
    x_k = -x_init;
    for k = 1:r
        % Find the tangent line.
        a_k = delta(x_k);
        b_k = -x_k.*delta(x_k)+log(exp(x_k)+1);
        approx(k,:) = [a_k, b_k];
        x(k) = x_k;
        x_k = x_k + eps;
    end
    approx(r+1,:) = [0, 0];
end

function [y] = delta(x)
    y = 1.-(1./(exp(x)+1));
end