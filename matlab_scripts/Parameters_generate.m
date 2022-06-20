% Creates a parameter vector where a discrete laplace kernel is positioned
% at the center of every location at all time instances in the memory
% horizon.
function theta = Parameters_generate(L, d, rows, cols, type, density)
    theta = zeros(L, d*L);
    switch type
        case 'laplace'
            dim = 5;
            k = ones(dim,dim).*(0.125);
            k(2:4,2) = [-0.125 -0.125 -0.125];
            k(2:4,3) = [-0.125 -1 -0.125];
            k(2:4,4) = [-0.125 -0.125 -0.125];
            theta = filter_pass(rows, cols, d, theta, k, dim);
        case 'gaussian'
            % Rotationally symmetric Gaussian lowpass filter.
            sigma = 3;
            dim = 5;
            k = fspecial('gaussian', [dim dim], sigma);
            theta = filter_pass(rows, cols, d, theta, k, dim);
        case '1-ball'
            theta = ballL1(n);
            theta = reshape(theta, L, d*L);
        otherwise
            theta = sprandn(1, L*d*L, density);
            theta = reshape(theta, L, d*L);
    end
end


function theta = filter_pass(rows, cols, d, theta, k, dim)
    sh = floor(dim/2);
    mid = ceil(dim/2);
    % Gaussian-filter like parameter vector.
    for i = 1:rows
        for j = 1:cols
            temp = zeros(rows, cols);
            % Left-upper corner.
            if i<=mid && j<=mid
                temp(1:(i+sh),1:(j+sh)) = k(mid-i+1:dim,mid-j+1:dim);
            % Right-upper corner.
            elseif i<=mid && j>=(cols-sh)
                temp(1:(i+sh),(j-sh):cols) = k(mid-i+1:dim,1:mid+(cols-j));
            % Left-bottom corner.
            elseif i>=(rows-sh) && j<=mid
                temp((i-sh):rows,1:(j+sh)) = k(1:mid+(rows-i),mid-j+1:dim);
            % Right-bottom corner.
            elseif i>=(rows-sh) && j>=(cols-sh)
                temp((i-sh):rows,(j-sh):cols) = k(1:mid+(rows-i),1:mid+(cols-j));
            % Upper edge.
            elseif i<=mid && j>mid && j<(cols-sh)
                temp(1:(i+sh),(j-sh):(j+sh)) = k(mid-i+1:dim,:);
            % Bottom edge.
            elseif i>=(rows-sh) && j>mid && j<(cols-sh)
                temp((i-sh):rows,(j-sh):(j+sh)) = k(1:mid+(rows-i),:);
            % Right edge.
            elseif i>mid && i<(rows-sh) && j>=(cols-sh)
                temp((i-sh):(i+sh),(j-sh):cols) = k(:,1:mid+(cols-j));
            % Left edge.
            elseif i>mid && i<(rows-sh) && j<=mid
                temp((i-sh):(i+sh),1:(j+sh)) = k(:,mid-j+1:dim);
            % Intermediate location.
            else
                temp((i-sh):(i+sh),(j-sh):(j+sh)) = k;
            end
            temp = reshape(temp.', 1, []);
            theta((i-1)*cols+j, :) = repmat(temp,1,d);
        end
    end
end