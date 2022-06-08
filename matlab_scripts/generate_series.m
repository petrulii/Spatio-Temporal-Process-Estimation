% Generate time series with d*periods+1 time steps.
function [time_horizon, N, L, theta, theta0, init_grids] = generate_series(rows, cols, d, periods, density)
    % Number of locations.
    L = rows*cols;
    N = d + d*periods;
    % Initialiazing the time horizon.
    init_grids = zeros(d,L);
    time_horizon = zeros(N,L);
    % Create a random Bernoulli process grid at the initial time strech.
    for s = (1:d)
        x = normrnd(0, 1, 1, L);
        x(x>=0) = 1;
        x(x<0) = 0;
        for l = 1:L
            time_horizon(s,:) = x;
        end
    end
    % Initialising the sparse true parameter vector and the initial probability.
    theta = sprandn(L, d*L, density);
    %theta(x~=0) = theta(x~=0)+0.5;
    %theta = zeros(L, d*L);
    % Cluster-inducing parameter vector.
    %{
for curr = 1:L
        for s = 1:d
            for l = 1:L
                if s==d && (l==curr-1 || l==curr+1 || l==curr-cols || l==curr+cols)
                    theta(curr, d, l) = 0.4;
                elseif s==d-1 && (l==curr-1 || l==curr+1 || l==curr-cols || l==curr+cols)
                    theta(curr, d, l) = 0.3;
                elseif s==d-2 && (l==curr-1 || l==curr+1 || l==curr-cols || l==curr+cols)
                    theta(curr, d, l) = 0.2;
                elseif s==d-3 && (l==curr-1 || l==curr+1 || l==curr-cols || l==curr+cols)
                    theta(curr, d, l) = 0.1;
                end
            end
        end
    end
    %}
    fprintf('%s\n %d\n', 'Part of non-zero values in the true parameter vector:', nnz(theta)/(d*L*L));
    theta0 = normrnd(0,1,1,L);
    % Generate time series.
    for s = (d+1):(N+1)
        % Predictor X of dimension d*L.
        X = time_horizon((s-d):(s-1),:);
        X = reshape(X.',1,[]);
        for l = 1:L
            % Train data.
            if s ~= (N+1)
                p = sigmoid(theta0(l) + dot(X, theta(l,:)));
                time_horizon(s,l) = Bernouilli_draw(p);
            % Test data.
            else
                time_horizon(s,l) = sigmoid(theta0(l) + dot(X, theta(l,:)));
            end
        end
    end
    fprintf('%s\n %d\n', 'Part of non-zero values in the time horizon:', nnz(time_horizon)/(N*L));
    fprintf('%s\n', 'Last values of the time horizon:');
    disp(time_horizon(N-2*d:N,L-8:L));
end

% Bernouilli draw with probability p.
function y = Bernouilli_draw(p)
    r = rand();
    if r <= p
        y = 1;
    else
        y = 0;
    end
end
