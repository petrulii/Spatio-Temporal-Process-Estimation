% Main function.
function CVX_regression_circular
    % Set the random seed.
    rng('default');
    % Memory depth.
    d = 3;
    % The length of the time horizon is d*periods+1.
    periods = 4;
    % Dimensions of 2-D space at any time instance.
    row = 7;
    col = row;
    % Norm for regularization (1-lasso, 2-ridge).
    reg = 1;
    % Lists for plotting
    error_log_l1 = [];
    error_lin_l1 = [];
    zer_log = [];
    zer_lin = [];
    theta_norm_log = [];
    theta_norm_lin = [];
    all_lambda = [];
    
    % Determine density.
    density = 0.05;
    % Generating Bernouilli time series of N time instances and L locations.
    [time_horizon, N, L, true_theta, true_b] = generate_series(row, col, d, periods, density);
        
    % Trying multiple lasso hyper-parameter values.
    for lambda = 0:0.000000025:0.000000075
        % LGR+LASSO : Logistic regression with lasso.
        [theta, b] = logistic(time_horizon, N, L, d, 1, lambda);
        % Generate a prediction and compare with groud truth.
        [err_log_l1, z_log, b_n_log] = predict(time_horizon((N-d)+1:N,:), time_horizon(N+1,:), L, d, true_theta, theta, b, row, col, @sigmoid, false);
        zer_log = [zer_log z_log];
        error_log_l1 = [error_log_l1 err_log_l1];
        theta_norm_log = [theta_norm_log b_n_log];
        
        % LNR+LASSO : Linear regression with lasso.
        [theta, b] = linear(time_horizon, N, L, d, 1, lambda);
        % Generate a prediction and compare with groud truth.
        [err_lin_l1, z_lin, b_n_lin] = predict(time_horizon((N-d)+1:N,:), time_horizon(N+1,:), L, d, true_theta, theta, b, row, col, @identity, false);
        zer_lin = [zer_lin z_lin];
        error_lin_l1 = [error_lin_l1 err_lin_l1];
        theta_norm_lin = [theta_norm_lin b_n_lin];
        
        all_lambda = [all_lambda lambda];
    end
    
    % Plotting.
    figure(2);
    hold on;
    plot(all_lambda, zer_log);
    plot(all_lambda, zer_lin);
    xlabel('Reguralization hyper-parameter \lambda');
    ylabel('Null values in \theta');
    legend('Log-regression + L1','Linear-regression + L1');
    saveas(gcf, 'null_values', 'png');
    hold off;
    figure(3);
    hold on;
    plot(all_lambda, error_log_l1);
    plot(all_lambda, error_lin_l1);
    xlabel('Reguralization hyper-parameter \lambda');
    ylabel('Prediction error at t+1');
    legend('Log-regression + L1','Linear-regression + L1');
    saveas(gcf, 'error', 'png');
    hold off;
    figure(4);
    hold on;
    plot(all_lambda, theta_norm_log);
    plot(all_lambda, theta_norm_lin);
    xlabel('Reguralization hyper-parameter \lambda');
    ylabel('2-norm of \theta - \theta^*');
    legend('Log-regression + L1','Linear-regression + L1');
    saveas(gcf, 'distance', 'png');
    hold off;
end

% Maximum likelihood estimation.
function [theta, init_intens] = logistic(time_horizon, N, L, d, reg, lambda)
        cvx_begin
            variable theta(d*L);
            variable init_intens(L);
            obj = 0;
            for s = d:N
                X = time_horizon((s-d+1):s,:);
                X = reshape(X.',1,[]);
                % For each location in the 2-D grid.
                for l = 1:L
                    y = time_horizon(s+1,l);
                    b = init_intens(l);
                    % Log-likelihood with L1 penalty.
                    obj = obj + (y*(dot(X,theta)+b) - log(1+exp(dot(X,theta)+b))) - lambda * norm(theta,reg);
                end
            end
            maximize(obj);
        cvx_end
        % Transform small values of theta to 0s.
        theta(theta>-0.0001 & theta<0.0001) = 0;
end

% Least-squares estimation.
function [theta, init_intens] = linear(time_horizon, N, L, d, reg, lambda)
        cvx_begin
            variable theta(d*L);
            variable init_intens(L);
            obj = 0;
            for s = d:N
                X = time_horizon((s-d+1):s,:);
                X = reshape(X.',1,[]);
                % For each location in the 2-D grid.
                for l = 1:L
                    y = time_horizon(s+1,l);
                    b = init_intens(l);
                    % Distance.
                    obj = obj + norm(y-(dot(X,theta)+b))/2 + lambda * norm(theta,reg);
                end
            end
            minimize(obj);
        cvx_end
        % Transform small values of theta to 0s.
        theta(theta>-0.0001 & theta<0.0001) = 0;
end

% Prediction for time series of 2-D Bernouilli events.
function [err, zer, dist] = predict(X, y, L, d, true_theta, theta, b, rows, columns, activation, heatmap)
    if ~exist('heatmap','var')
        heatmap = false; end
    X = reshape(X.',1,[]);
    y = reshape(y.',1,[]);
    prediction = normrnd(0,1,1,L);
    obj = 0;
    norm1 = 0;
    zer = 0;
    for l = 1:L
        prediction(l) = activation(b(l) + dot(X, theta));
        obj = obj + (y(l)*(dot(X,theta)+b(l)) - log(1+exp(dot(X,theta)+(l))));
    end
    zer = zer + sum(theta==0);
    % Error of the prediction.
    err = immse(y,prediction);
    % 2-norm of the difference btween estimated theta and true theta.
    dist = norm((true_theta-theta),2);
    % Null values in the estimated theta.
    zer = zer/(d*L*L);
    fprintf('%s %d %s %d %s %d %s %d %s %d\n', 'Prediction error:', err, 'distance between estimation and true parameters:', dist, 'zero values:', zer, 'likelihood:', obj, 'l1 norm:', norm1);
    fprintf('%s\n', 'First values of estimated theta:');
    disp(theta(1:3));
    fprintf('%s\n', 'First values of true theta:');
    disp(true_theta(1:3));
    % Plot the ground truth and prediction heatmaps.
    if heatmap == true
        y = reshape(y,rows,columns);
        colormap('hot');
        imagesc(y);
        colorbar;
        fig1 = gcf;
        saveas(fig1,'ground_truth','png');
        prediction = reshape(prediction,rows,columns);
        colormap('hot');
        imagesc(prediction);
        colorbar;
        fig2 = gcf;
        saveas(fig2,'prediction','png');
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
    theta = sprandn(1, d*L, density);
    fprintf('%s\n %d\n', 'Part of non-zero values in the true parameter vector:', nnz(theta)/(d*L));
    fprintf('%s\n', 'First values of the parameter vector:');
    disp(theta(1:2*d));
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
                time_horizon(s,l) = binary_sigmoid(b(l) + dot(X, theta));
            % Test data.
            else
                time_horizon(s,l) = binary_sigmoid(b(l) + dot(X, theta));
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