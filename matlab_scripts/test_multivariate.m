function [] = test_multivariate()
    n = 1;
    r = 10;
    eps = 0.7;
    x_init = ones(n,1).*(r*eps/2);
    [approx, x] = approximate(n, r, eps, x_init);
    disp(x);
    disp(approx);
    y = zeros(1,r);
    for k = 1:r
        y(k) = find_sup(r, approx, x(k));
    end
    plot(x,y);
end

function [y_max] = find_sup(r, approx, x)
    y_max = -1000;
    for k = 1:r
        a_k = approx(k,1);
        b_k = approx(k,2);
        y = hyperplane(x, a_k, b_k);
        if y>y_max
            y_max = y;
        end
    end
end

function [approx, x] = approximate(n, r, eps, x_init)
    approx = zeros(r,n+1);
    x = zeros(r,n);
    x_k = -x_init;
    for k = 1:r
        % Find the tangent hyperplane
        approx(k,:) = find_hyperplane(x_k, n);
        x(k,:) = x_k;
        x_k = x_k+eps;
    end
end

function [y] = lse(x)
    y = log(1+exp(x));
end

function [w] = find_hyperplane(a, n)
    w = zeros(1,n+1);
    for i=1:n
        w(i) = find_a(a, n, i);
    end
    w(n+1) = find_b(a, n);
end

function [a_i] = find_a(a, n, i)
    sum = 1;
    for j=1:n
        sum = sum + exp(a(j));
    end
    a_i = exp(a(i))/(sum);
end

function [b] = find_b(a, n)
    sum = 1;
    % Calculate the denominator.
    for j=1:n
        sum = sum + exp(a(j));
    end
    res = 0;
    for j=1:n
        res = res - a(j)*exp(a(j))/sum;
    end
    b = res + log(sum);
end

function y = hyperplane(x, a, b)
    y = a.*x+b;
end