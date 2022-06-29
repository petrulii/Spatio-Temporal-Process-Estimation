
function [] = test()
    % Set the random seed.
    %rng(0);
    % Memory depth.
    d = 2;
    % Dimensions of 2-D space at any time instance.
    rows = 2;
    cols = rows;
    x = zeros(1,1000);
    y = zeros(1,1000);
    for i=1:1000
        v = ball_uniform(2);
        x(i) = v(1);
        y(i) = v(2);
    end
    % Draw one vector from a 2-ball.
    density = 0.5;
    theta = ball_operator(rows, cols, d, density);
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