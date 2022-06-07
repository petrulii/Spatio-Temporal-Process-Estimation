% Main function.
function Parameter_recovery
    % Set the random seed.
    rng(0);
    % The length of the time horizon is d*periods+1.
    periods = 15;
    % Dimensions of 2-D space grid.
    row = 4;
    col = row;
    % Density of the true parameter vector.
    density = 0.1;
    % Lists for plotting
    error_log_l1 = [];
    error_lin_l1 = [];
    error_clin_l1 = [];
    zer_log_l1 = [];
    zer_lin_l1 = [];
    zer_clin_l1 = [];
    theta_norm_log_l1 = [];
    theta_norm_lin_l1 = [];
    theta_norm_clin_l1 = [];
    theta_mse_log_l1 = [];
    theta_mse_lin_l1 = [];
    theta_mse_clin_l1 = [];
    % Memory depths.
    all_depths = linspace(3,6,4);
    all_lambdas = logspace(-6,1,8);
    disp(all_lambdas);
    all_dens = linspace(0.1,0.4,4);
    all_periods = linspace(2,16,8);
    % Memeory depth.
    d = 3;
    % Reguralization parameter.
    lbd = 1;
    
    % Regularization hyper-parameter.
    for lbd = all_lambdas
    
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
        
        % LNR+LASSO : Linear regression with lasso.
        [theta, theta0] = linear(time_horizon, N, L, d, lbd);
        % Generate a prediction and compare with groud truth.
        [err_lin_l1, z_lin_l1, t_n_lin_l1, t_ms_lin_l1] = predict(time_horizon((N-d)+1:N,:), time_horizon(N+1,:), L, d, true_theta, theta, true_theta0, theta0, row, col, @identity);
        zer_lin_l1 = [zer_lin_l1 z_lin_l1];
        error_lin_l1 = [error_lin_l1 err_lin_l1];
        theta_norm_lin_l1 = [theta_norm_lin_l1 t_n_lin_l1];
        theta_mse_lin_l1 = [theta_mse_lin_l1 t_ms_lin_l1];
        
        % LNR+CONSTRAINTS : Linear regression with constraints.
        [theta, theta0] = linear_constraints(time_horizon, N, L, d, lbd);
        % Generate a prediction and compare with groud truth.
        [err_clin_l1, z_clin_l1, t_n_clin_l1, t_ms_clin_l1] = predict(time_horizon((N-d)+1:N,:), time_horizon(N+1,:), L, d, true_theta, theta, true_theta0, theta0, row, col, @identity);
        zer_clin_l1 = [zer_clin_l1 z_clin_l1];
        error_clin_l1 = [error_clin_l1 err_clin_l1];
        theta_norm_clin_l1 = [theta_norm_clin_l1 t_n_clin_l1];
        theta_mse_clin_l1 = [theta_mse_clin_l1 t_ms_clin_l1];
        
    end
    
    result_plot(log(all_lambdas), zer_log_l1, error_log_l1, theta_norm_log_l1, theta_mse_log_l1, zer_lin_l1, error_lin_l1, theta_norm_lin_l1, theta_mse_lin_l1, zer_clin_l1, error_clin_l1, theta_norm_clin_l1, theta_mse_clin_l1);
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
            obj = obj - lambda * sum(sum(abs(theta)));
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
            obj = obj / (N*L) + lambda * sum(sum(abs(theta)));
            minimize(obj);
        cvx_end
        % Transform small values of theta to 0s.
        theta(theta>-0.0001 & theta<0.0001) = 0;
end

% Least-squares estimation.
function [theta, init_intens] = linear_constraints(time_horizon, N, L, d, lambda)
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
                    subject to
                        dot(X,a)+b <= 1;
                        dot(X,a)+b >= 0;
                end
            end
            obj = obj / (N*L) + lambda * sum(sum(abs(theta)));
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

% Log-it activation function.
function y = sigmoid(x)
    y = (1+exp(-x)).^(-1);
end

% Identity activation function.
function y = identity(x)
    y = x;
end

% Binary log-it activation function.
function y = Bernouilli_draw(p)
    r = rand();
    if r <= p
        y = 1;
    else
        y = 0;
    end
end

% Generate time series with d*periods+1 time steps.
function [time_horizon, N, L, theta, theta0] = generate_series(L_rows, L_columns, d, periods, density)
    % Number of locations.
    L = L_rows*L_columns;
    N = d + d*periods;
    % Initialiazing the time horizon.
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
    fprintf('%s\n %d\n', 'Part of non-zero values in the true parameter vector:', nnz(theta)/(d*L*L));
    theta0 = sprandn(1, L, density);
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

function [] = series_plot(L, N, d, theta_true, theta0_true, theta, theta0, density)
    % For plotting.
    N = N+(N/2);
    norm_X_true = zeros(1,N);
    norm_probability_true = zeros(1,N);
    norm_X = zeros(1,N);
    norm_probability = zeros(1,N);
    % Initialiazing the time horizon.
    time_horizon = zeros(N,L);
    time_horizon_true = zeros(N,L);
    % Create a random Bernoulli process grid at the initial time strech.
    for s = (1:d)
        x = sprand(L, 1, density);
        x(x>0) = 1;
        for l = 1:L
            time_horizon(s,:) = x;
            time_horizon_true(s,:) = x;
        end
    end
    % Generate time series.
    for s = (d+1):(N)
        % Predictor X of dimension d*L.
        X = time_horizon((s-d):(s-1),:);
        X = reshape(X.',1,[]);
        X_true = time_horizon_true((s-d):(s-1),:);
        X_true = reshape(X_true.',1,[]);
        for l = 1:L
            p = sigmoid(theta0(l) + dot(X, theta(l,:)));
            time_horizon(s,l) = Bernouilli_draw(p);
            p_true = sigmoid(theta0_true(l) + dot(X_true, theta_true(l,:)));
            time_horizon_true(s,l) = Bernouilli_draw(p_true);
            % Calculating the 2-norms.
            norm_probability(1,s) = norm_probability(1,s) + p^2;
            norm_X(1,s) = norm_X(1,s) + time_horizon(s,l);
            norm_probability_true(1,s) = norm_probability_true(1,s) + p_true^2;
            norm_X_true(1,s) = norm_X_true(1,s) + time_horizon_true(s,l);
        end
    end
    % Plotting
    time = 1:N;
    base_proba = sigmoid(theta0);
    norm_theta0 = repelem(sqrt(sum(base_proba.^2)), N);
    norm_probability = sqrt(norm_probability);
    norm_X = sqrt(norm_X);
    base_proba_true = sigmoid(theta0_true);
    norm_theta0_true = repelem(sqrt(sum(base_proba_true.^2)), N);
    norm_probability_true = sqrt(norm_probability_true);
    norm_X_true = sqrt(norm_X_true);
    f0 = figure('visible','on');
    hold on;
    plot(time, norm_theta0);
    plot(time, norm_probability);
    plot(time, norm_X);
    plot(time, norm_theta0_true);
    plot(time, norm_probability_true);
    plot(time, norm_X_true);
    xlabel('Time t');
    ylabel('2-norm');
    title('True and estimated time series');
    legend('estimated \beta_0 at time t','estimated \beta at time t','predicted Y_t at time t', 'true \beta_0^* at time t','true \beta^* at time t','true Y_t^* at time t');
    saveas(f0, 'predicted_true_series', 'png');
    hold off;
end

function [] = result_plot(x_values, zer_log_l1, error_log_l1, theta_norm_log_l1, theta_mse_log_l1, zer_lin_l1, error_lin_l1, theta_norm_lin_l1, theta_mse_lin_l1, zer_clin_l1, error_clin_l1, theta_norm_clin_l1, theta_mse_clin_l1)
    f1 = figure('visible','on');
    hold on;
    plot(x_values, zer_log_l1);
    plot(x_values, zer_lin_l1);
    plot(x_values, zer_clin_l1);
    title('Non-zeros in the estimated parameter vector \beta');
    xlabel('Log-scale lasso penalty hyper-parameter \lambda');
    %xlabel('Memeory depth d');
    %xlabel('Length of the series N');
    ylabel('Non-zero values in \beta');
    legend('Maximum likelihood + L1','Least squares + L1', 'Least squares + L1 + const.');
    saveas(f1, 'null_values_theta', 'png');
    hold off;
    f2 = figure('visible','on');
    hold on;
    plot(x_values, log(error_log_l1));
    plot(x_values, log(error_lin_l1));
    plot(x_values, log(error_clin_l1));
    title('Prediction error of the response vector at t+1');
    xlabel('Log-scale lasso penalty hyper-parameter \lambda');
    ylabel('Log-scale MSE of Y_{t+1}^{*} and Y_{t+1}');
    legend('Maximum likelihood + L1','Least squares + L1', 'Least squares + L1 + const.');
    saveas(f2, 'error_prediction', 'png');
    hold off;
    f3 = figure('visible','on');
    hold on;
    plot(x_values, theta_norm_log_l1);
    plot(x_values, theta_norm_lin_l1);
    plot(x_values, theta_norm_clin_l1);
    title('Prediction error of the true par. vector');
    xlabel('Log-scale lasso penalty hyper-parameter \lambda');
    ylabel('2-norm of \beta^* - \beta');
    legend('Maximum likelihood + L1','Least squares + L1', 'Least squares + L1 + const.');
    saveas(f3, 'distance_theta', 'png');
    hold off;
    f4 = figure('visible','on');
    hold on;
    plot(x_values, log(theta_mse_log_l1));
    plot(x_values, log(theta_mse_lin_l1));
    plot(x_values, log(theta_mse_clin_l1));
    title('MSE of the estimated parameter vector');
    xlabel('Log-scale lasso penalty hyper-parameter \lambda');
    ylabel('Log-scale MSE of \beta^* and \beta');
    legend('Maximum likelihood + L1','Least squares + L1', 'Least squares + L1 + const.');
    saveas(f4, 'mse_theta', 'png');
    hold off;
end

function [] = color_plot(v_true,v_pred)
    bottom = min(min(min(v_true)),min(min(v_pred)));
    top  = max(max(max(v_true)),max(max(v_pred)));
    f = figure('visible','off');
    % Plotting the first plot
    sp1 = subplot(1,2,1);
    colormap('hot');
    imagesc(v_true);
    xlabel(sp1, 'true \beta^*');
    shading interp;
    % This sets the limits of the colorbar to manual for the first plot
    caxis manual;
    caxis([bottom top]);
    colorbar;
    % Plotting the second plot
    sp2 = subplot(1,2,2);
    colormap('hot');
    imagesc(v_pred);
    xlabel(sp2, 'estimated \beta');
    shading interp;
    % This sets the limits of the colorbar to manual for the second plot
    caxis manual;
    caxis([bottom top]);
    colorbar;
    saveas(f, 'color_maps', 'png');
end
