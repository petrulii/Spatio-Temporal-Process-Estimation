% Creates a parameter vector where a discrete laplace kernel is positioned
% at the center of every location at all time instances in the memory
% horizon.
function theta = laplace2D(d, L, rows, cols)
    theta = zeros(L, d*L);
    % Cluster-inducing parameter vector.
    for i = 1:rows
        for j = 1:cols
            temp = zeros(rows, cols);
            temp(i,j) = -1;
            % Left-upper corner.
            if i==1 && j==1
                temp(i+1,j+1) = 1/3;
                temp(i,j+1) = 1/3;
                temp(i+1,j) = 1/3;
            % Right-upper corner.
            elseif i==1 && j==cols
                temp(i+1,j-1) = 1/3;
                temp(i,j-1) = 1/3;
                temp(i+1,j) = 1/3;
            % Left-bottom corner.
            elseif i==rows && j==1
                temp(i-1,j+1) = 1/3;
                temp(i,j+1) = 1/3;
                temp(i-1,j) = 1/3;
            % Right-bottom corner.
            elseif i==rows && j==cols
                temp(i-1,j-1) = 1/3;
                temp(i,j-1) = 1/3;
                temp(i-1,j) = 1/3;
            % Upper edge.
            elseif i==1
                temp(i+1,j) = 1/3;
                temp(i,j-1) = 1/3;
                temp(i,j+1) = 1/3;
            % Left edge.
            elseif j==1
                temp(i,j+1) = 1/3;
                temp(i-1,j) = 1/3;
                temp(i+1,j) = 1/3;
            % Bottom edge.
            elseif i==rows
                temp(i-1,j) = 1/3;
                temp(i,j-1) = 1/3;
                temp(i,j+1) = 1/3;
            % Right edge.
            elseif j==cols
                temp(i,j-1) = 1/3;
                temp(i-1,j) = 1/3;
                temp(i+1,j) = 1/3;
            % Intermediate location.
            else
                temp(i,j-1) = 1/4;
                temp(i,j+1) = 1/4;
                temp(i-1,j) = 1/4;
                temp(i+1,j) = 1/4;
            end
            temp = reshape(temp.', 1, []);
            theta((i-1)*cols+j, :) = repmat(temp,1,d);
        end
    end
end