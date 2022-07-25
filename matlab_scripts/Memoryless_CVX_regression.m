% Main function.
function Memoryless_CVX_regression
    % Set the random seed.
    rng('default');
    % Memory depth.
    d = 3;
    % The length of the time horizon is d*periods+1.
    periods = 6;
    N = periods*d;
    % Dimensions of 2-D space at any time instance.
    row = 6;
    col = row;
    L = row*col;
    % Norm for regularization (1-lasso, 2-ridge).
    reg = 1;
    % Lists for plotting
    error_log_l1 = [];
    error_lin_l1 = [];
    zer_log = [];
    zer_lin = [];
    all_lambda = [];
    % Trying multiple lasso hyper-parameter values.
    for lambda = 0%:0.000000025:0.00000005%0.0000005
        % Determine density.
        density = (d)/(periods*row*col);
        fprintf('%s %d %s %d\n', 'Initial density of the series:', density, 'lambda:', lambda);
        % Initialising the true parameter vector and the bias.
        %true_theta = normrnd(0, 1, L, d*L);
        true_theta = normrnd(-1, 1, L, d*L);
        true_b = normrnd(0, 1, 1, L);
        % LGR+LASSO : Logistic regression with lasso.
        [theta, b, X] = logistic(density, N, L, d, true_theta, true_b, reg, lambda);
        % Generate a prediction and compare with groud truth.
        [err_log_l1, zlog] = predict(X, L, d, true_theta, true_b, theta, b, row, col, @sigmoid, true);
        error_log_l1 = [error_log_l1 err_log_l1];
        zer_log = [zer_log zlog];
        % LNR+LASSO : Linear regression with lasso.
        [theta, b, X] = linear(density, N, L, d, true_theta, true_b, reg, lambda);
        % Generate a prediction and compare with groud truth.
        [err_lin_l1, zlin] = predict(X, L, d, true_theta, true_b, theta, b, row, col, @identity, true);
        error_lin_l1 = [error_lin_l1 err_lin_l1];
        zer_lin = [zer_lin zlin];
        all_lambda = [all_lambda lambda];
    end
    % Plotting.
    figure(1);
    hold on;
    plot(all_lambda, zer_log);
    plot(all_lambda, zer_lin);
    xlabel('\lambda');
    ylabel('Null values in \theta');
    legend('Log-regression + L1','Linear-regression + L1');
    saveas(gcf, 'memoryless_null_values_log', 'png');
    hold off;
    figure(2);
    hold on;
    plot(all_lambda, error_log_l1);
    plot(all_lambda, error_lin_l1);
    xlabel('\lambda');
    ylabel('Prediction error at t+1');
    legend('Log-regression + L1','Linear-regression + L1');
    saveas(gcf, 'memoryless_error', 'png');
    hold off;
end

% Maximum likelihood estimation.
function [theta, bias, X] = logistic(density, N, L, d, true_theta, true_b, reg, lambda)
        X = generate_series(L, d, 0, density, 1, true_theta, true_b);
        cvx_begin
            variable theta(L, d*L);
            variable bias(L);
            obj = 0;
            for s = d:N
                X = generate_series(L, d, X, 0, s-d+1, true_theta, true_b);
                X = reshape(X.',1,[]);
                % For each location in the 2-D grid.
                for l = 1:L
                    a = theta(l,:);
                    b = bias(l);
                    y = binary_sigmoid(true_b(l) + dot(X, true_theta(l,:)));
                    % Log-likelihood with L1 penalty.
                    obj = obj + (y*(dot(X,a)+b) - log(1+exp(dot(X,a)+b))) - lambda * norm(a,reg);
                end
            end
            maximize(obj);
        cvx_end
        % Transform small values of theta to 0s.
        theta(theta>-0.0001 & theta<0.0001) = 0;
end

% Least-squares estimation.
function [theta, bias, X] = linear(density, N, L, d, true_theta, true_b, reg, lambda)
        X = generate_series(L, d, 0, density, 1, true_theta, true_b);
        cvx_begin
            variable theta(L, d*L);
            variable bias(L);
            obj = 0;
            for s = d:N
                X = generate_series(L, d, X, 0, s-d+1, true_theta, true_b);
                X = reshape(X.',1,[]);
                % For each location in the 2-D grid.
                for l = 1:L
                    a = theta(l,:);
                    b = bias(l);
                    y = binary_sigmoid(true_b(l) + dot(X, true_theta(l,:)));
                    % Distance.
                    obj = obj + norm(y-(dot(X,a)+b))/2 + lambda * norm(a,reg);
                end
            end
            minimize(obj);
        cvx_end
        % Transform small values of theta to 0s.
        theta(theta>-0.0001 & theta<0.0001) = 0;
end

% Prediction for time series of 2-D Bernouilli events.
function [err, zer] = predict(X, L, d, true_theta, true_b, theta, b, rows, columns, activation, heatmap)
    if ~exist('heatmap','var')
        heatmap = false; end
    prediction = normrnd(0,1,1,L);
    y = normrnd(0,1,1,L);
    obj = 0;
    norm1 = 0;
    zer = 0;
    for l = 1:L
        y(l) = sigmoid(true_b(l) + dot(X, true_theta(l,:)));
        prediction(l) = activation(b(l) + dot(X, theta(l,:)));
        a = theta(l,:);
        obj = obj + (y(l)*(dot(X,a)+b(l)) - log(1+exp(dot(X,a)+(l))));
        norm1 = norm1 + norm(a,1);
        zer = zer + sum(a==0);
    end
    % Error of the prediction.
    err = immse(y,prediction);
    % 2-norm of the difference btween estimated theta and true theta.
    dist = norm((true_theta-theta),2);
    % Null values in the estimated theta.
    zer = zer/(d*L*L);
    fprintf('%s %d %s %d %s %d %s %d %s %d\n', 'Prediction error:', err, 'distance between estimation and true parameters:', dist, 'zero values:', zer, 'likelihood:', obj, 'l1 norm:', norm1);
    fprintf('%s\n', 'First values of estimated theta:');
    disp(theta(1,1:3));
    fprintf('%s\n', 'First values of true theta:');
    disp(true_theta(1,1:3));
    % Plot the ground truth and prediction heatmaps.
    if heatmap == true
        y = reshape(y,rows,columns);
        colormap('hot');
        imagesc(y);
        colorbar;
        saveas(gcf,'ground_truth','png');
        prediction = reshape(prediction,rows,columns);
        colormap('hot');
        imagesc(prediction);
        colorbar;
        saveas(gcf,'prediction','png');
    end
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

% Generate time series with d*periods+1 time steps,
% where the value at time t is an L_rows*L_columns
% binary matrix of specified density.
function X = generate_series(L, d, X_prev, density, low, theta, b)
    % Initialiazing the time horizon.
    X = zeros(d,L);
    % Create a random Bernoulli process grid at the initial time strech.
    if low == 1
        for s = (1:d)
            x = sprand(L, 1, density);
            x(x>0) = 1;
            for l = 1:L
                X(s,:) = x;
            end
        end
    else
    % Generate time series at some intermediate time strech.
    for s = 1:d
        for l = 1:L
            X(s,l) = binary_sigmoid(b(l) + dot(X_prev, theta(l,:)));
        end
    end
    end
end