% Main function.
function horizon_tests
    % Set the random seed.
    rng(0);
    % Memory depth.
    d = 3;
    % The length of the time horizon is d*periods+1.
    periods = 200;
    % Dimensions of 2-D space at any time instance.
    row = 10;
    col = row;
    % Trying multiple lasso hyper-parameter values.
    % Determine density.
    density = 0.044;
    % Generating Bernouilli time series of N time instances and L locations.
    [time_horizon, N, L, true_theta, true_b] = generate_series(row, col, d, periods, density);
end

% Log-it activation function.
function y = sigmoid(x)
    y = 1/(1+exp(-x));
end

% Identity activation function.
function y = identity(x)
    y = x;
end

% Binary log-it activation function.
function y = binary_sigmoid(x)
    if (1/(1+exp(-x))) >= 0.5
        y = 1;
    else
        y = 0;
    end
end

function [time_horizon, N, L, theta, b] = generate_series(L_rows, L_columns, d, periods, density)
    % Number of locations.
    L = L_rows*L_columns;
    % Creating random events with some density over an n by m grid.
    N = d + d*periods;
    % For plotting.
    norm_X = zeros(1,N+1);
    % Initialiazing the time horizon.
    time_horizon = zeros(N,L);
    % Create a random Bernoulli process grid at the initial time strech.
    for s = (1:d)
        x = sprand(L, 1, density);
        x(x>0) = 1;
        norm_X(s) = norm(x,2);
        for l = 1:L
            time_horizon(s,:) = x;
        end
    end
    % Initialising the sparse true parameter vector and the initial probability.
    %theta = normrnd(0, 1, L, d*L);
    theta = sprandn(L, d*L, density);
    fprintf('%s\n %d\n', 'Part of non-zero values in the true parameter vector:', nnz(theta)/(d*L*L));
    fprintf('%s\n', 'First values of the parameter vector:');
    disp(theta(1:2*d,1:2*d));
    % Putting half of the true parameter vector values below 0.
    b = normrnd(0, 1, 1, L);
    % Generate time series.
    for s = (d+1):(N+1)
        % Predictor X of dimension d*L.
        X = time_horizon((s-d):(s-1),:);
        X = reshape(X.',1,[]);
        norm_X(1,s) = norm(X,2);
        for l = 1:L
            % Train data.
            if s ~= (N+1)
                time_horizon(s,l) = binary_sigmoid(b(l) + dot(X, theta(l,:)));
            % Test data.
            else
                time_horizon(s,l) = binary_sigmoid(b(l) + dot(X, theta(l,:)));
            end
        end
    end
    fprintf('%s\n %d\n', 'Part of non-zero values in the time horizon:', nnz(time_horizon)/(N*L));
    fprintf('%s\n', 'Last values of the time horizon:');
    disp(time_horizon(N-2*d:N,L-8:L));
    % Plotting
    time = 1:(N+1);
    norm_b = repelem(norm(b,2), N+1);
    figure(1);
    hold on;
    plot(time, norm_b);
    plot(time, norm_X);
    xlabel('Time t');
    ylabel('2-norm');
    legend('Initial intensity','Time horizon');
    saveas(gcf, 'time_horizon_vs_initial_intensities', 'png');
    hold off;
end