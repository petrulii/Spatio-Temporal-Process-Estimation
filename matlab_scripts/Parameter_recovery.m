% Main function.
function Parameter_recovery
    % Set the random seed.
    %rng(1);
    % The length of the time horizon is d*periods+1.
    periods = 30;
    % Dimensions of 2-D space grid.
    row = 6;
    col = row;
    % Memory depths.
    all_lambdas = logspace(-4,4,20);
    len_lambdas = length(all_lambdas);
    % Memeory depth.
    d = 3;
    % Lists for plotting
    iterations = 10;
    error_log_l1 = zeros(iterations,len_lambdas);
    error_lin_l1 = zeros(iterations,len_lambdas);
    zer_log_l1 = zeros(iterations,len_lambdas);
    zer_lin_l1 = zeros(iterations,len_lambdas);
    theta_norm_log_l1 = zeros(iterations,len_lambdas);
    theta_norm_lin_l1 = zeros(iterations,len_lambdas);

    for i = 1:iterations
        % Regularization hyper-parameter.
        for j = 1:len_lambdas

            lbd = all_lambdas(j);
            % Generating Bernouilli time series of N+1 time instances and L locations.
            %radius = d;
            %value = 1;
            [time_series, probabilities, N, L, true_theta, true_theta0] = generate_series(row, col, d, periods, 'operator', 0, 0);

            % LGR+LASSO : Logistic regression with lasso.
            [theta, theta0] = logistic(time_series, N, L, d, lbd);
            % Generate a prediction and compare with groud truth.
            [err_log_l1, z_log_l1, t_n_log_l1] = predict(time_series((N-d)+1:N,:), time_series(N+1,:), L, d, true_theta, theta, true_theta0, theta0, row, col, @sigmoid);
            zer_log_l1(i,j) = z_log_l1;
            error_log_l1(i,j) = err_log_l1;
            theta_norm_log_l1(i,j) = t_n_log_l1;

            % LNR+LASSO : Linear regression with lasso.
            [theta, theta0] = linear(time_series, N, L, d, lbd);
            % Generate a prediction and compare with groud truth.
            [err_lin_l1, z_lin_l1, t_n_lin_l1] = predict(time_series((N-d)+1:N,:), time_series(N+1,:), L, d, true_theta, theta, true_theta0, theta0, row, col, @identity);
            zer_lin_l1(i,j) = z_lin_l1;
            error_lin_l1(i,j) = err_lin_l1;
            theta_norm_lin_l1(i,j) = t_n_lin_l1;

        end
    end
    
    Parameter_recovery_plot(all_lambdas, zer_log_l1, error_log_l1, theta_norm_log_l1, zer_lin_l1, error_lin_l1, theta_norm_lin_l1);
end

% Maximum likelihood estimation.
function [theta, init_intens] = logistic(time_series, N, L, d, lambda)
        cvx_solver mosek;
        cvx_begin;
            variable theta(L, d*L);
            variable init_intens(L);
            obj = 0;
            for s = d:(N-1)
                X = time_series((s-d+1):s,:);
                X = reshape(X.',1,[]);
                % For each location in the 2-D grid.
                for l = 1:L
                    y = time_series(s+1,l);
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
function [theta, init_intens] = linear(time_series, N, L, d, lambda)
        cvx_begin
            variable theta(L, d*L);
            variable init_intens(L);
            obj = 0;
            for s = d:(N-1)
                X = time_series((s-d+1):s,:);
                X = reshape(X.',1,[]);
                % For each location in the 2-D grid.
                for l = 1:L
                    y = time_series(s+1,l);
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
function [err, zer, dist] = predict(X, y, L, d, true_theta, theta, true_theta0, theta0, rows, columns, activation, heatmap)
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
    zer = nnz(theta);
    % Null values in the estimated theta.
    zer = zer/(d*L*L);
    fprintf('%s %d\n', 'length of estimated theta:', length(theta));
    fprintf('%s %d\n', 'length of true theta:', length(true_theta));
    fprintf('%s %d\n', 'mean of true theta:', mean(true_theta));
    fprintf('%s %d\n', '2-norm of true theta:', norm(true_theta,2));
    fprintf('%s %d\n', '2-norm of estimation difference:', dist);
end

% Identity activation function.
function y = identity(x)
    y = x;
end
