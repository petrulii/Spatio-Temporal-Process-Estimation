% Generate time series with d*periods+1 time steps.
function [time_series, probabilities, N, L, theta, theta0] = generate_series(rows, cols, d, periods, density, type)
    % Number of locations.
    L = rows*cols;
    N = d + d*periods;
    % Initialiazing the time horizon.
    time_series = zeros(N,L);
    probabilities = zeros(N,L);
    % Create a random Bernoulli process grid at the initial time strech.
    for s = (1:d)
        x = normrnd(0, 1, 1, L);
        x(x>=0) = 1;
        x(x<0) = 0;
        %time_series(s,:) = x;  % Random initial grids.
        
        for l = 1:L
            if l>(L/2)
                time_series(s,l) = 0;
            else
                time_series(s,l) = 1;
            end
        end
        
    end
    % Initialising the sparse true parameter vector and the initial probability.
    theta = Parameters_generate(L, d, rows, cols, type, density);
    fprintf('%s\n %d\n', 'Part of non-zero values in the true parameter vector:', nnz(theta)/(d*L*L));
    theta0 = normrnd(0,1,1,L);
    % Generate time series.
    for s = (d+1):(N+1)
        % Predictor X of dimension d*L.
        X = time_series((s-d):(s-1),:);
        X = reshape(X.',1,[]);
        for l = 1:L
            % Train data.
            if s ~= (N+1)
                p = sigmoid(theta0(l) + dot(X, theta(l,:)));
                time_series(s,l) = Bernouilli_draw(p);
                probabilities(s,l) = p;
            % Test data.
            else
                time_series(s,l) = sigmoid(theta0(l) + dot(X, theta(l,:)));
            end
        end
    end
    fprintf('%s\n %d\n', 'Part of non-zero values in the time horizon:', nnz(time_series)/(N*L));
    fprintf('%s\n', 'Last values of the time horizon:');
    disp(time_series(N-2*d:N,L-8:L));
end
