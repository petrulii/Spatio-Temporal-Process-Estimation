function [] = Parameter_recovery_mirror_prox()
    % Set the random seed.
    rng(0);
    % Dimensions of 2-D space grid.
    row = 4;
    col = row;
    % Memeory depth.
    d = 2;
    periods = 500;
    % Values used in parameter generation.
    radius = 1;
    values = [1 -1];
    % Generating Bernouilli time series of N+1 time instances and L locations.
    [time_series, probabilities, N, L, true_theta, true_theta0] = generate_series(row, col, d, periods, 'random', radius, values);
    
    kappa = 1;
    rate = 0.1;
    max_iterations = 1000;
    [theta, theta0, y] = mirror_prox(N, L, d, time_series, kappa, rate, max_iterations, true_theta, true_theta0);
    series_plot(L, N, d, true_theta, true_theta0, theta, theta0);
end

function [theta, theta0, y] = mirror_prox(N, L, d, time_series, kappa, rate, max_iterations, true_theta, true_theta0)
    % Extragradient descent.
    % param x_init: initial strategy vector for player X
    % param y_init: initial strategy vector for player Y
    theta = ones(L,d*L);
    theta0 = ones(1,L);
    y = 1;

    log_loss_error = zeros(1,max_iterations);
    estimation_error = zeros(1,max_iterations);
    prediction_error = zeros(1,max_iterations);
    i = 1;

    while i <= max_iterations
        
        % Gradient step to go to an intermediate point.
        [theta_grad, theta0_grad] = gradient_theta(N, L, d, time_series, theta, theta0, y);
        y_grad = gradient_y(N, L, d, time_series, theta, theta0, kappa);
        
        % Calculate y_i.
        theta_ = projection(theta - rate*(theta_grad));
        theta0_ = projection(theta0 - rate*(theta0_grad));
        y_ = projsplx(y + rate*(y_grad));

        % Use the gradient of the intermediate point to perform a gradient step.
        [theta_grad_, theta0_grad_] = gradient_theta(N, L, d, time_series, theta, theta0, y_);
        y_grad_ = gradient_y(N, L, d, time_series, theta_, theta0_, kappa);
        
        % Calculate x_i+1.
        theta = projection(theta - rate*(theta_grad_));
        theta0 = projection(theta0 - rate*(theta0_grad_));
        y = projsplx(y + rate*(y_grad_));

        % Calculate the prediction error over the time horizon.
        [err, dist] = prediction(time_series((N-d)+1:N,:), time_series(N+1,:), L, true_theta, theta, true_theta0, theta0);
        log_loss_error(i) = neg_log_loss(N, L, d, time_series, theta, theta0);
        estimation_error(i) = dist;
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

% Identity function.
function y = projection(x)
    y = x;
end

% Gradient of the objective w.r.t. the parameter vector x of the process.
function [theta_grad, theta0_grad] = gradient_theta(N, L, d, time_series, theta, theta0, y)
    [theta_grad, theta0_grad] = log_loss_gradient(N, L, d, time_series, theta, theta0);
    theta_grad = theta_grad*y(1);
    theta0_grad = theta0_grad*y(1);
end

% Gradient of the objective w.r.t. the weigth vector y of the obj. f-ion and constraints.
function y_grad = gradient_y(N, L, d, time_series, theta, theta0, kappa)
    y_grad = neg_log_loss(N, L, d, time_series, theta, theta0) - kappa;
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
function res = l1_penalty(theta, theta0)
    res = sum(sum(abs(theta))) + sum(abs(theta0));
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