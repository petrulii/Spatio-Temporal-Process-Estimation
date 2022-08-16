function [] = piecewise_lin_approx()
    r = 20;
    %x_init = r*eps/2;
    [A, b] = battlse(r);%lse_lin_approx(r, eps, x_init);
    n = 20;
    x = linspace(-5,5,n);
    y = zeros(1,n);
    for i = 1:20
        %y(i) = max(A*x(i)+b);
        disp(size(A));
        x_ = [0 x(i)];
        disp(size(x_.'));
        y(i) = max(A*x_.'+b);
    end
    figure('visible','on');
    hold on;
    plot(x,y, '.');
    plot(x, lse(x), '-');
    title('Piecewise-Linear Approximation');
    xlabel('x');
    ylabel('f(x)');
    legend('Approximation','log(1+exp(x))');
    hold off;
end

function y = lse(x)
    y = log(1+exp(x));
end