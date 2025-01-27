function [] = Parameter_recovery_mirror_prox()
    % Set the random seed.
    rng(0);
    % Dimensions of 2-D space grid.
    row = 5;
    col = row;
    % Memeory depth.
    d = 2;
    % The length of the time horizon is d*periods+1.
    all_periods = [10 100 1000 10000];
    len_periods = length(all_periods);
    % Values used in parameter generation.
    radius = 1;
    values = [-1 -1];
    % Lists for plotting.
    iterations = 10;
    all_N = zeros(1, len_periods);
    error_log_l1 = zeros(iterations,len_periods);
    error_lin_l1 = zeros(iterations,len_periods);
    zer_log_l1 = zeros(iterations,len_periods);
    zer_lin_l1 = zeros(iterations,len_periods);
    theta_norm_log_l1 = zeros(iterations,len_periods);
    theta_norm_lin_l1 = zeros(iterations,len_periods);
    % Mirror-prox hyper-parameters.
    rate = 1;
    lambda = 0.0005;
    max_iterations = 1500;
    max_iterations_kappa = 10;
    
    [time_series, probabilities, N, L, true_theta, true_theta0] = generate_series(row, col, d, 10000, 'operator', radius, values);
    [theta, theta0] = estimate_parameters(N, L, d, time_series, rate, max_iterations, true_theta, true_theta0, lambda, max_iterations_kappa);
    return;
    
    for i = 1:iterations
        % Regularization hyper-parameter.
        for j = 1:len_periods
            
            fprintf('\n\n\n%s %d %s %d\n\n', 'Iteration:', i, ', period index:', j);
            periods = all_periods(j);
            % Generating Bernouilli time series of N+1 time instances and L locations.
            [time_series, probabilities, N, L, true_theta, true_theta0] = generate_series(row, col, d, periods, 'operator', radius, values);
            all_N(j) = N;
            
            % Maximum likelihood estimation with lasso.
            [theta, theta0] = estimate_parameters(N, L, d, time_series, rate, max_iterations, true_theta, true_theta0, lambda, max_iterations_kappa);
            % Generate a prediction and compare with groud truth.
            [err_log_l1, z_log_l1, t_n_log_l1] = predict(time_series((N-d)+1:N,:), time_series(N+1,:), L, d, true_theta, theta, true_theta0, theta0, row, col, @sigmoid);
            zer_log_l1(i,j) = z_log_l1;
            error_log_l1(i,j) = err_log_l1;
            theta_norm_log_l1(i,j) = t_n_log_l1;
            
        end
    end
    
    Parameter_recovery_plot(all_N, zer_log_l1, error_log_l1, theta_norm_log_l1, zer_lin_l1, error_lin_l1, theta_norm_lin_l1, 'N');
end

function [theta, theta0] = estimate_parameters(N, L, d, time_series, rate, max_iterations, true_theta, true_theta0, lambda, max_iterations_kappa)
    kappa = 1;
    [theta, theta0, y, obj] = mirror_prox(N, L, d, time_series, kappa, rate, max_iterations, true_theta, true_theta0, lambda);
    %theta = randn(L,d*L);
    %theta0 = randn(1,L);
    %{
    for i = 1:max_iterations_kappa
        fprintf('%s %d\n', 'kappa search iteration :', i);
        [theta, theta0, y, obj] = mirror_prox(N, L, d, time_series, kappa, rate, max_iterations, true_theta, true_theta0, lambda);
        if obj > 0
            prev_kappa = kappa;
            kappa = kappa * 2;
        else
            while 1
                if obj <= 0
                    %diff = (kappa - prev_kappa)/2;
                    prev_theta = theta;
                    prev_theta0 = theta0;
                    prev_kappa = kappa;
                    kappa = kappa - 0.5;%diff;
                    [theta, theta0, y, obj] = mirror_prox(N, L, d, time_series, kappa, rate, max_iterations, true_theta, true_theta0, lambda);
                else
                    theta = prev_theta;
                    theta0 = prev_theta0;
                    flag=1; 
                    break;
                end
            end
            if flag==1
                break;
            end
        end
    end
    %}
end

function [theta, theta0, y, obj] = mirror_prox(N, L, d, time_series, kappa, rate, max_iterations, true_theta, true_theta0, lambda)
    % Mirror prox.
    % param N: length of the time horizon of the time series
    % param L: number of locations in the 2D spatial grid
    % param d: memory depth of the process
    % param time_series: time series of the process
    theta = randn(L,d*L);
    theta0 = randn(1,L);
    y = [0 0];

    log_loss_error = zeros(1,max_iterations);
    estimation_error = zeros(1,max_iterations);
    prediction_error = zeros(1,max_iterations);
    i = 1;

    while i <= max_iterations

        fprintf('%s %d\n', 'Iteration :', i);
        fprintf('%s %d\n', 'kappa :', kappa);
        fprintf('%s %d %d\n', 'y0 y1 :', y(1), y(2));
        fprintf('%s %d\n', 'F_0(x) :', neg_log_loss(N, L, d, time_series, theta, theta0) + lambda * l1_norm(theta) - kappa);
        fprintf('%s %d\n', 'F_1(x) :', constraint1(theta));
        obj = F_kappa(N, L, d, time_series, theta, theta0, y, kappa, lambda);
        fprintf('%s %d\n', 'F(kappa) :', obj);
        fprintf('%s %d\n', 'neg log l. :', neg_log_loss(N, L, d, time_series, theta, theta0));
        fprintf('%s %d\n', 'sum(theta) :', sum(sum(theta)));
        fprintf('%s %d\n', 'sum(true_theta) :', sum(sum(true_theta)));
        fprintf('%s \n \t %d %d %d \n \t %d %d %d \n', 'First values of true theta :', true_theta(1,1), true_theta(1,2), true_theta(1,3), true_theta(2,1), true_theta(2,2), true_theta(2,3));
        fprintf('%s \n \t %d %d %d \n \t %d %d %d \n', 'First values of pred. theta :', theta(1,1), theta(1,2), theta(1,3), theta(2,1), theta(2,2), theta(2,3));

        % Gradient step to go to an intermediate point.
        [theta_grad, theta0_grad] = gradient_theta(N, L, d, time_series, theta, theta0, y, lambda);
        y_grad = gradient_y(N, L, d, time_series, theta, theta0, kappa, lambda);

        % Calculate y_i.
        theta_ = theta - rate*(theta_grad);
        theta0_ = theta0 - rate*(theta0_grad);
        % Project onto the simplex.
        y_ = projsplx(y + rate*(y_grad));

        % Use the gradient of the intermediate point to perform a gradient step.
        [theta_grad_, theta0_grad_] = gradient_theta(N, L, d, time_series, theta, theta0, y_, lambda);
        y_grad_ = gradient_y(N, L, d, time_series, theta_, theta0_, kappa, lambda);

        % Calculate x_i+1.
        theta = theta - rate*(theta_grad_);
        theta0 = theta0 - rate*(theta0_grad_);
        % Project onto the simplex.
        y = projsplx(y + rate*(y_grad_));
        theta(theta>-0.001 & theta<0.001) = 0;

        [err, dist] = prediction(time_series((N-d)+1:N,:), time_series(N+1,:), L, true_theta, theta, true_theta0, theta0);
        % Total Neg. Log Loss.
        log_loss_error(i) = neg_log_loss(N, L, d, time_series, theta, theta0);
        % Prediction error over the time horizon.
        estimation_error(i) = dist;
        % Error of estimation of the parameter vector of the process.
        prediction_error(i) = err;

        i = i + 1;
    end
    figure('visible','on');
    hold on;
    plot(log_loss_error);
    plot(estimation_error);
    plot(prediction_error);
    title('Error');
    xlabel('Iteration');
    ylabel('Error');
    legend('Negative Log Loss','Estimation error', 'Prediction error');
    hold off;
end

% Gradient of the objective w.r.t. the parameter vector x of the process.
function res = F_kappa(N, L, d, time_series, theta, theta0, y, kappa, lambda)
    res = y(1) * ((neg_log_loss(N, L, d, time_series, theta, theta0) + lambda * l1_norm(theta)) - kappa) + y(2) * constraint1(theta);
end

% Gradient of the objective w.r.t. the parameter vector x of the process.
function [theta_grad, theta0_grad] = gradient_theta(N, L, d, time_series, theta, theta0, y, lambda)
    [theta_grad, theta0_grad] = log_loss_gradient(N, L, d, time_series, theta, theta0);
    theta_grad = (theta_grad + l1_prox(theta, L, d*L, lambda))*y(1) + y(2);
    theta0_grad = (theta0_grad + l1_prox(theta0, 1, L, lambda))*y(1) + y(2);
end

function res = l1_prox(x, rows, cols, lambda)
    res = zeros(rows, cols);
    for i = 1:rows
        for j = 1:cols
            if x(i,j) > 0
                res(i,j) = 1 * lambda;%max(abs(x(i,j))-lambda, 0);
            elseif x(i,j) < 0
                res(i, j) = -1 * lambda;% * max(abs(x(i,j))-lambda, 0);
            end
        end
    end
end

% Gradient of the objective w.r.t. the weigth vector y of the obj. f-ion and constraints.
function y_grad = gradient_y(N, L, d, time_series, theta, theta0, kappa, lambda)
    y1_grad = neg_log_loss(N, L, d, time_series, theta, theta0) + lambda * l1_norm(theta) - kappa;
    y2_grad = constraint1(theta);
    y_grad = [y1_grad, y2_grad];
end

% Gradient descent for time series of 2-D Bernouilli events.
function [theta_grad, theta0_grad] = log_loss_gradient(N, L, d, series, theta, theta0)
    theta_grad = zeros(L,d*L);
    theta0_grad = zeros(1,L);
    % For each time instance in the time horizon from d to N.
    for s = d:(N-1)
        % Take values from the last d time instances.
        X = series((s-d+1):s,:);
        X = reshape(X.',1,[]);
        % For each location in the 2D grid of the current time instance.
        for l = 1:L
            y = series(s+1,l);
            a = theta(l,:);
            b = theta0(l);
            % Update the parameter vector.
            theta_grad(l,:) = theta_grad(l,:) + X.*((-1)/(exp(a*X.'+b) + 1)-y+1);
            theta0_grad(l) = theta0_grad(l) + ((-1)/(exp(a*X.'+b) + 1)-y+1);
        end
    end
    theta_grad = theta_grad./((N-d-1)*L);
    theta0_grad = theta0_grad./((N-d-1)*L);
        
end

% Negative cross-entropy loss.
function obj = neg_log_loss(N, L, d, time_series, theta, theta0)
    obj = 0;
    for s = d:(N-1)
        X = time_series((s-d+1):s,:);
        % TODO: try without the following and with transpose
        X = reshape(X.',1,[]);
        % For each location in the 2-D grid.
        for l = 1:L
            y = time_series(s+1,l);
            a = theta(l,:);
            b = theta0(l);
            % Log-likelihood with L1 penalty.
            obj = obj + (log_sum_exp([0; (dot(X,a)+b)]) - (y*(dot(X,a)+b)));
            %obj = obj + log(1 + exp(X.'*theta+theta0)) - y*(X.'*theta+theta0) / (N*L);
        end
    end
    obj = obj/((N-d)*L);
end

% L1 penalty function.
function res = l1_norm(theta)
    res = sum(sum(abs(theta)));
end

function f1 = constraint1(theta)
    f1 = sum(sum(theta));
end

function [rate] = linesearch_stepsize(x_i, y_i, x_i_1, grad_y_i, x2_i, y2_i, x2_i_1, grad_y2_i, rate)
    % Backtrack line search for step size.
    i=0;
    while i<2
        if (rate*np.dot((grad_y_i.T),(y_i-x_i_1)) <= (1/2)*np.power(norm(x_i-x_i_1, 2),2)) && (rate*dot((grad_y2_i.T),(y2_i-x2_i_1)) <= (1/2)*power(norm(x2_i-x2_i_1, 2),2))
            beta = np.sqrt(2);
        else
            beta = 0.5;
        end
        rate = rate * beta;
        i=i+1;
    end
end

% Prediction for time series of 2-D Bernouilli events.
function [err, dist] = prediction(X, y, L, true_theta, theta, true_theta0, theta0)
    X = reshape(X.',1,[]);
    prediction = zeros(1,L);
    % For each location in the 2-D grid.
    for l = 1:L
        prediction(l) = sigmoid(theta0(l) + dot(X, theta(l,:)));
    end
    % Squared error of the prediction.
    err = immse(y, prediction);
    % Squared error btween estimated theta and true theta.
    true_theta = full(true_theta);
    true_theta0 = full(true_theta0);
    true_theta = reshape(true_theta.',1,[]);
    true_theta = [true_theta true_theta0];
    theta = reshape(theta.',1,[]);
    theta = [theta theta0];
    dist = sqrt(sum((true_theta-theta).^2));
end