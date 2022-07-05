function [] = test()
    r = 40;
    eps = 0.2;
    approx = approximate(r, eps);
    figure('visible','on');
    hold on;
    syms x;
    for k = 1:r
        a_k = approx(k,1);
        b_k = approx(k,2);
        fplot(hyperplane(x, a_k, b_k));
    end
    fplot(lse(x));
    hold off;
end

function [approx] = approximate(r, eps)
    %x_low = log(exp(eps)-1);
    %x_high = -log(exp(eps)-1);
    approx = zeros(r,2);
    x_k = -2;%x_low;
    for k = 1:r
        % Find the tangent hyperplane
        [a_k, b_k] = find_hyperplane(x_k);
        disp('x, a, b:');
        disp(x_k);
        disp(a_k);
        disp(b_k);
        approx(k,:) = [a_k, b_k];
        x_k = x_k + eps;
    end
end

function [y] = lse(x)
    y = log(1+exp(x));
end

function [y] = delta(x)
    y = 1.-(1./(exp(x)+1));
end

function [a_k, b_k] = find_hyperplane(a)
    a_k = delta(a);
    b_k = -a.*delta(a)+log(exp(a)+1);
end

function y = hyperplane(x, a, b)
    y = a.*x+b;
end

%{
function [] = test()
    r = 20;
    approx = approximate(r, eps);
    disp(approx);
    x = linspace(-5,5,r);
    y = zeros(1,r);
    for k = 1:r
        y(k) = find_sup(r, approx, x(k));
    end
    plot(x,y);
end

function [y_max] = find_sup(r, approx, x)
    y_max = -1;
    for k = 1:r
        a_k = approx(k,1);
        b_k = approx(k,2);
        y = hyperplane(x, a_k, b_k);
        if y>y_max
            y_max = y;
        end
    end
end

function [approx] = approximate(r, eps)
    %x_low = log(exp(eps)-1);
    %x_high = -log(exp(eps)-1);
    approx = zeros(r,2);
    x_k = -5;%x_low;
    for k = 1:r
        % Find the tangent hyperplane
        [a_k, b_k] = find_hyperplane(x_k);
        approx(k,:) = [a_k, b_k];
        x_k = x_k + eps;
    end
end

function [y] = lse(x)
    y = log(1+exp(x));
end

function [y] = delta(x)
    y = 1.-(1./(exp(x)+1));
end

function [a_k, b_k] = find_hyperplane(a)
    a_k = delta(a);
    b_k = 0;%a.*delta(a)+log(exp(a)+1);
end

function y = hyperplane(x, a, b)
    y = a.*x+b;
end
    
function [theta] = ball_operator(rows, cols, d, density)
    L = rows*cols;
    theta = zeros(L, d*L);
    for i = 1:rows
        for j = 1:cols
            l = (i-1)*cols+j;
            alpha = ball_uniform(d*L);
            disp(alpha);
            disp(sum(alpha.^2));
            x = (sprandn(1, L*d, density));
            x(x>0) = 1;
            temp = x.*alpha;
            theta(l, :) = temp;
            disp(theta(l, :));
            disp(sum(theta(l, :).^2));
        end
    end
end
%}