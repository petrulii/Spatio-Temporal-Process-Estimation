
function theta = filter_pass(rows, cols, d, theta, k, dim)
    sh = floor(dim/2);
    mid = ceil(dim/2);
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

% Identity function.
function y = identity(x)
    y = x;
end

% Mean squared error.
function err = MSE(series, theta, N, L, d)
    err = 0;
    for s = d+1:N
        X = series((s-d):(s-1),:);
        X = reshape(X.',1,[]);
        for l = 1:L
            y = series(s,l);
            y_pred = theta(l,:)*X.'+theta0;
            err = err + mean(sumsqr(y - y_pred));
        end
    end
    err = err/N;
end

% L1 norm of vector x of dimension n.
function y = lasso(x, n, lambda)
    y = x;
    penalty = norm(y,1);
    for i = 1:n
        if y(i) < 0
            y(i) = y(i) + lambda * penalty;
        elseif y(i) > 0
            y(i) = y(i) - lambda * penalty;
        end
    end
end