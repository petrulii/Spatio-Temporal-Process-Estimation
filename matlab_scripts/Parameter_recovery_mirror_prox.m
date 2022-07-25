function [] = Parameter_recovery_mirror_prox()
    % Set the random seed.
    rng(0);
    % Dimensions of 2-D space grid.
    row = 4;
    col = row;
    % Memeory depth.
    d = 2;
    periods = 10;
    % Values used in parameter generation.
    radius = 1;
    values = [1 -1];
    % Generating Bernouilli time series of N+1 time instances and L locations.
    [time_series, probabilities, N, L, true_theta, true_theta0] = generate_series(row, col, d, periods, 'operator', radius, values);

    theta_init = zeros(L, d*L);
    theta0_init = zeros(1, L);
    y_init = zeros(1, 2);
    rate = 0.1;
    target_error = 100;
    disp(mirror_prox(N, L, d, time_series, theta_init, theta0_init, y_init, rate, target_error));
end

function [x, y] = mirror_prox(N, L, d, time_series, theta_init, theta0_init, y_init, rate, target_error)
    % Extragradient descent.
    % param x_init: initial strategy vector for player X
    % param y_init: initial strategy vector for player Y
    %error = 1000;
    theta = theta_init;
    theta0 = theta0_init;
    y = y_init;
    i = 0;

    if i < 2%error > target_error
        
        % Gradient step to go to an intermediate point.
        theta_grad = gradient_theta(N, L, d, time_series, theta, theta0, y);
        y_grad = gradient_y(N, L, d, time_series, theta, theta0, kappa);
        
        % Calculate y_i.
        theta_ = projection(theta - rate*(theta_grad));
        y_ = projection(y - rate*(y_grad));

        % Use the gradient of the intermediate point to perform a gradient step.
        theta_grad_ = gradient_theta(N, L, d, time_series, theta, theta0, y_);
        y_grad_ = gradient_y(N, L, d, time_series, theta_, theta0_, kappa);
        
        % Calculate x_i+1.
        theta = projection(theta - rate*(theta_grad_));
        y = projection(y - rate*(y_grad_));
        
        i=i+1;
    end
end

% Identity function.
function y = projection(x)
    y = x;
end

% Gradient of the objective w.r.t. the parameter vector x of the process.
function theta_grad = gradient_theta(N, L, d, time_series, theta, theta0, y)
    theta_grad = y(1)*log_loss_gradient(N, L, d, time_series, theta, theta0) + y(2)*k*l1_penalty(theta, theta0);
end

% Logistic loss gradient w.r.t. the parameter vector.
function theta_log_loss_grad = log_loss_gradient(N, L, d, time_series, theta, theta0)
    theta_log_loss_grad = zeros(L, d*L);
    %theta0_log_loss_grad = zeros(1, L);
    for s = d:(N-1)
        X = time_series((s-d+1):s,:);
        % TODO: try without the following and with transpose
        X = reshape(X.',1,[]);
        for l = 1:L
            y = time_series(s+1,l);
            a = theta(l,:);
            b = theta0(l);
            disp(size(X.'));
            disp(size(((exp(X.'*a+b)/(exp(X.'*a+b)+1))-y)));
            theta_log_loss_grad(l,:) = theta_log_loss_grad(l,:) + X.'*((exp(X.'*a+b)/(exp(X.'*a+b)+1))-y);
        end
    end
end

% Gradient of the objective w.r.t. the weigth vector y of the obj. f-ion and constraints.
function y_grad = gradient_y(N, L, d, time_series, theta, theta0, kappa)
    F0_theta = neg_log_loss(N, L, d, time_series, theta, theta0);
    disp(F0_theta);
    y_grad = F0_theta + k*l1_penalty(theta, theta0) - kappa;
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