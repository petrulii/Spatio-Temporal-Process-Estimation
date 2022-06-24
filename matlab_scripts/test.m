function [] = test()
    % Set the random seed.
    rng(0);
    % Memory depth.
    d = 3;
    % Dimensions of 2-D space at any time instance.
    row = 5;
    col = row;
    s = 3;
    % Draw one vector from a 2-ball.
    alpha = ball_uniform(2, 1);
    disp(alpha);
    a1 = (alpha(1)-alpha(2))^s;
    disp(a1);
    a2 = exp(alpha(1)-alpha(2));
    disp(a2);
    radius = 2;
    sobel_operator(row, col, d, radius,1);
end

function [theta] = sobel_operator(rows, cols, d, radius, s)
    L = rows*cols;
    theta = zeros(L, d*L);
    for i = 1:rows
        for j = 1:cols
            temp = zeros(rows, cols);
              for x = max(1, i-radius):min(i+radius, rows)
                for y = max(1, j-radius):min(j+radius, cols)
                  temp(x,y) = (x-y)^s;
                end
              end
            disp(temp);
            temp = reshape(temp.', 1, []);
            theta((i-1)*cols+j, :) = repmat(temp,1,d);
        end
    end
end







% Generates the parameter vector of different types.
function theta = Parameters_generate(L, d, rows, cols, type, val)
    switch type
        % The parameter vector is an operator of specified neighbourhood.
        case 'operator'
            center = val;
            values = [val val -val -val -val];
            initial = 0;
            theta = operator_pass(rows, cols, d, center, values, initial, length(values));
        % The parameter vector is a sobel-like operator constructed from coordinates.
        case 'sobel'
            radius = 2;
            theta = sobel_operator(rows, cols, d, radius, 1);
        case 'd-ball'
        otherwise
            density = 0.2;
            theta = sprandn(1, L*d*L, density);
            theta = reshape(theta, L, d*L);
    end
end

% Creates a parameter vector where a discrete laplace kernel is positioned
% at the center of every location at all time instances in the memory
% horizon.
function [theta] = operator_pass(rows, cols, d, center, values, initial, radius)
    theta = zeros(L, d*L);
    for i = 1:rows
        for j = 1:cols
            temp = ones(rows, cols).*initial;
            temp(i,j)=center;
            for r=1:radius
                n = count_neighbours(i,j,rows,cols,r);
                temp = assign_values(i,j,rows,cols,r,values(r)/n,temp);
            end
            temp = reshape(temp.', 1, []);
            theta((i-1)*cols+j, :) = repmat(temp,1,d);
        end
    end
end

function n = count_neighbours(i,j,rows,cols,layer)
    n = all_neighbours(i,j,rows,cols,layer)-all_neighbours(i,j,rows,cols,layer-1);
end

function c = all_neighbours(i,j,rows,cols,dist)
    c = 0;
    if dist~=0
        if(rows > 1)
          for x = max(1, i-dist):min(i+dist, rows)
            for y = max(1, j-dist):min(j+dist, cols)
              if(x ~= i || y ~= j)
                c=c+1;
              end
            end
          end
        end
    end
end

function temp = assign_values(i,j,rows,cols,r,value,temp)
    if r~=0
        if(rows > 1)
          for x = max(1, i-r):min(i+r, rows)
            for y = max(1, j-r):min(j+r, cols)
              if(x ~= i || y ~= j)
                if(x==(i+r) || x==(i-r) || y==(j+r) || y==(j-r))
                    temp(x,y)=value;
                end
              end
            end
          end
        end
    end
end

function [theta] = sobel_operator(rows, cols, d, radius, s)
    theta = zeros(L, d*L);
    for i = 1:rows
        for j = 1:cols
            temp = zeros(rows, cols);
              for x = max(1, i-radius):min(i+radius, rows)
                for y = max(1, j-radius):min(j+radius, cols)
                  temp(x,y) = (x-y)^s;
                end
              end
            disp(temp);
            temp = reshape(temp.', 1, []);
            theta((i-1)*cols+j, :) = repmat(temp,1,d);
        end
    end
end