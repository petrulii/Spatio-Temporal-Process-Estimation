% Main function.
function time_series_utils
    % Setting the random seed.
    rng('default')
    % Setting the parameters of 2-D binary time series.
    row = 4;
    col = row;
    rate = 0.2;
    max_iterations = 1000;
    d = 2;
    periods = 500;
    % Values used in parameter generation.
    radius = 1;
    values = [1 -1];
    % Generating Bernouilli time series of N+1 time instances and L locations.
    [time_series, probabilities, N, L, true_theta, true_theta0] = generate_series(row, col, d, periods, 'operator', radius, values);
    % Inferring the parameter vector of the time series.
    [theta, theta0, log_loss_error, estimation_error, prediction_error] = estimate_parameters(time_series, N, L, d, rate, max_iterations, true_theta, true_theta0);
    % Plotting the training error over all iterations.
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
function [theta_grad, theta0_grad] = gradient_theta(N, L, d, time_series, theta, theta0, y)
    [theta_grad, theta0_grad] = log_loss_gradient(N, L, d, time_series, theta, theta0);
    theta_grad = theta_grad*y(1);
    theta0_grad = theta0_grad*y(1);
end

% Gradient descent for time series of 2-D Bernouilli events.
function [theta, theta0, log_loss_error, estimation_error, prediction_error] = estimate_parameters(series, N, L, d, rate, max_iterations, true_theta, true_theta0)
    %{
    param series : a time series of 2-D categorical events
    param N : lenght of the time series
    param L : number of locations in the 2-D grid where categorical events take place
    param d : memory depth describing the depth of the autoregression
    param rate : learning rate for gradient descent
    param max_error : maximum tolerated error
    param max_iterations : maximum tolerated number of gradient steps
    %}
    theta = ones(L,d*L);
    theta0 = ones(1,L);
    log_loss_error = zeros(1,max_iterations);
    estimation_error = zeros(1,max_iterations);
    prediction_error = zeros(1,max_iterations);
    i = 1;

    while (i <= max_iterations)
        theta_grad = zeros(L,d*L);
        theta0_grad = zeros(1,L);
        fprintf('%s %d\n', 'Iteration :', i);
        fprintf('%s %d %d %d\n', 'First values of true theta :', true_theta(1), true_theta(2), true_theta(3));
        fprintf('%s %d %d %d\n', 'First values of pred. theta :', theta(1,1), theta(1,2), theta(1,3));
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
        theta = theta - rate * theta_grad./((N-d-1)*L);
        theta0 = theta0 - rate * theta0_grad./((N-d-1)*L);
        % Calculate the prediction error over the time horizon.
        [err, dist] = prediction(series((N-d)+1:N,:), series(N+1,:), L, true_theta, theta, true_theta0, theta0);
        log_loss_error(i) = neg_log_loss(N, L, d, series, theta, theta0);
        estimation_error(i) = dist;
        prediction_error(i) = err;
        
        i = i+1;
    end
end

% Negative cross-entropy loss.
function obj = neg_log_loss(N, L, d, time_series, theta, theta0)
    obj = 0;
    for s = d:(N-1)
        X = time_series((s-d+1):s,:);
        X = reshape(X.',1,[]);
        % For each location in the 2-D grid.
        for l = 1:L
            y = time_series(s+1,l);
            a = theta(l,:);
            b = theta0(l);
            % Log-likelihood.
            obj = obj + (log_sum_exp([0; (a*X.'+b)]) - (y*(a*X.'+b)));
        end
    end
    obj = obj/((N-d)*L);
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