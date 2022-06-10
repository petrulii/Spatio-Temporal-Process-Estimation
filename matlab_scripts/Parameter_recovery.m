% Main function.
function Parameter_recovery
    % Set the random seed.
    rng(1);
    % The length of the time horizon is d*periods+1.
    periods = 16;
    % Dimensions of 2-D space grid.
    row = 4;
    col = row;
    % Density of the true parameter vector.
    density = 0.35;
    % Lists for plotting
    error_log_l1 = [];
    error_lin_l1 = [];
    zer_log_l1 = [];
    zer_lin_l1 = [];
    theta_norm_log_l1 = [];
    theta_norm_lin_l1 = [];
    theta_mse_log_l1 = [];
    theta_mse_lin_l1 = [];
    % Memory depths.
    all_depths = linspace(3,6,4);
    all_lambdas = logspace(-4,4,100);
    all_dens = linspace(0.1,0.4,4);
    % Memeory depth.
    d = 3;

    % Regularization hyper-parameter.
    for lbd = 0
        
        % Generating Bernouilli time series of N+1 time instances and L locations.
        [time_horizon, N, L, true_theta, true_theta0] = generate_series(row, col, d, periods, density);
        
        % LGR+LASSO : Logistic regression with lasso.
        [theta, theta0] = logistic(time_horizon, N, L, d, lbd);
        % Generate a prediction and compare with groud truth.
        [err_log_l1, z_log_l1, t_n_log_l1, t_ms_log_l1] = predict(time_horizon((N-d)+1:N,:), time_horizon(N+1,:), L, d, true_theta, theta, true_theta0, theta0, row, col, @sigmoid);
        zer_log_l1 = [zer_log_l1 z_log_l1];
        error_log_l1 = [error_log_l1 err_log_l1];
        theta_norm_log_l1 = [theta_norm_log_l1 t_n_log_l1];
        theta_mse_log_l1 = [theta_mse_log_l1 t_ms_log_l1];
        
        location_plot(L, N, d, true_theta, true_theta0, theta, theta0, 2, time_horizon(1:d,:));
        return;
    
        % LNR+LASSO : Linear regression with lasso.
        [theta, theta0] = linear(time_horizon, N, L, d, lbd);
        % Generate a prediction and compare with groud truth.
        [err_lin_l1, z_lin_l1, t_n_lin_l1, t_ms_lin_l1] = predict(time_horizon((N-d)+1:N,:), time_horizon(N+1,:), L, d, true_theta, theta, true_theta0, theta0, row, col, @identity);
        zer_lin_l1 = [zer_lin_l1 z_lin_l1];
        error_lin_l1 = [error_lin_l1 err_lin_l1];
        theta_norm_lin_l1 = [theta_norm_lin_l1 t_n_lin_l1];
        theta_mse_lin_l1 = [theta_mse_lin_l1 t_ms_lin_l1];
        
    end
    
    result_plot(all_lambdas, zer_log_l1, error_log_l1, theta_norm_log_l1, theta_mse_log_l1, zer_lin_l1, error_lin_l1, theta_norm_lin_l1, theta_mse_lin_l1);
end

% Maximum likelihood estimation.
function [theta, init_intens] = logistic(time_horizon, N, L, d, lambda)
        %cvx_solver mosek;
        cvx_begin;
            variable theta(L, d*L);
            variable init_intens(L);
            obj = 0;
            for s = d:(N-1)
                X = time_horizon((s-d+1):s,:);
                X = reshape(X.',1,[]);
                % For each location in the 2-D grid.
                for l = 1:L
                    y = time_horizon(s+1,l);
                    a = theta(l,:);
                    b = init_intens(l);
                    % Log-likelihood with L1 penalty.
                    obj = obj + (y*(dot(X,a)+b) - log_sum_exp([0; (dot(X,a)+b)]));
                end
            end
            obj = obj - lambda * (sum(sum(abs(theta))) + sum(abs(init_intens)));
            maximize(obj);
        cvx_end;
        % Transform small values of theta to 0s.
        theta(theta>-0.0001 & theta<0.0001) = 0;
end

% Least-squares estimation.
function [theta, init_intens] = linear(time_horizon, N, L, d, lambda)
        cvx_begin
            variable theta(L, d*L);
            variable init_intens(L);
            obj = 0;
            for s = d:(N-1)
                X = time_horizon((s-d+1):s,:);
                X = reshape(X.',1,[]);
                % For each location in the 2-D grid.
                for l = 1:L
                    y = time_horizon(s+1,l);
                    a = theta(l,:);
                    b = init_intens(l);
                    % Distance.
                    obj = obj + (y-(dot(X,a)+b))^2;
                end
            end
            obj = obj / (N*L) + lambda * (sum(sum(abs(theta))) + sum(abs(init_intens)));
            minimize(obj);
        cvx_end
        % Transform small values of theta to 0s.
        theta(theta>-0.0001 & theta<0.0001) = 0;
end

% Prediction for time series of 2-D Bernouilli events.
function [err, zer, dist, theta_mse] = predict(X, y, L, d, true_theta, theta, true_theta0, theta0, rows, columns, activation, heatmap)
    fprintf('%s\n', 'First values of estimated theta:');
    disp(theta(1:2,1:8));
    fprintf('%s\n', 'First values of true theta:');
    disp(true_theta(1:2,1:8));
    if ~exist('heatmap','var')
        heatmap = false; end
    % Plot the true theta and predicted theta heatmaps.
    if heatmap == true
        true_th = reshape(true_theta(2:4,:),d*3,L);
        th = reshape(theta(2:4,:),d*3,L);
        color_plot(true_th, th);
    end
    X = reshape(X.',1,[]);
    y = reshape(y.',1,[]);
    prediction = normrnd(0,1,1,L);
    for l = 1:L
        prediction(l) = activation(theta0(l) + dot(X, theta(l,:)));
    end
    % Squared error of the prediction.
    err = immse(y, prediction);
    % Squared error btween estimated theta and true theta.
    true_theta = full(true_theta);
    true_theta0 = full(true_theta0);
    true_theta = reshape(true_theta.',1,[]);
    true_theta = [true_theta true_theta0];
    theta = reshape(theta.',1,[]);
    theta = [theta transpose(theta0)];
    dist = sqrt(sum((true_theta-theta).^2));
    theta_mse = immse(true_theta, theta);
    zer = nnz(theta);
    % Null values in the estimated theta.
    zer = zer/(d*L*L);
    fprintf('%s %d\n', 'length of estimated theta:', length(theta));
    fprintf('%s %d\n', 'mean of estimated theta:', mean(theta));
    fprintf('%s %d\n', '2-norm of estimated theta:', norm(theta,2));
    fprintf('%s %d\n', 'length of true theta:', length(true_theta));
    fprintf('%s %d\n', 'mean of true theta:', mean(true_theta));
    fprintf('%s %d\n', '2-norm of true theta:', norm(true_theta,2));
    fprintf('%s %d\n', '2-norm of estimation difference:', dist);
    fprintf('%s %d\n', 'MSE of estimation difference:', immse(theta, true_theta));
end

% Identity activation function.
function y = identity(x)
    y = x;
end
