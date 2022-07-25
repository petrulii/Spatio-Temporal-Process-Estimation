% Main function.
function main
    % Setting the random seed.
    rng('default')
    % Setting the parameters of 2-D binary time series.
    rows = 6;
    columns = 5;
    memory_depth = 3;
    periods = 4;
    rate = 0.1;
    density = 0.01;
    loss = @log_loss;
    activation = @sigmoid;
    gradient = @log_loss_gradient;
    reg = @lasso;
    reg_param = 0.001;
    % Generating time series of 2-D Bernouilli events.
    [time_horizon, N, L, true_theta] = generate_series(rows, columns, memory_depth, periods, density);
    % Inferring the parameter vector of the time series.
    [theta, i, log_err] = estimate_parameters(time_horizon, N, L, memory_depth, rate, 0.01, 600, activation, loss, gradient);
    similarity = mae(theta-true_theta);
    fprintf('%s %d %s %d %s %d\n', 'Number of iterations:', i, 'similarity:', similarity);
    % Plotting the training error over all iterations.
    plot(log_err);
    % Setting parameters for prediction.
    X_test = time_horizon((N-memory_depth)+1:N,:);
    X_test = reshape(X_test.',1,[]);
    % Generate a prediction and compare with groud truth.
    predict(X_test, L, true_theta, theta, rows, columns, activation);
end

% Prediction for time series of 2-D Bernouilli events.
function prediction = predict(X_test, L, true_theta, theta, rows, columns, activation, heatmap)
    if ~exist('heatmap','var')
        heatmap = false; end
    y = normrnd(0,1,L,1);
    prediction = normrnd(0,1,L,1);
    for l = 1:L
        % Calculate the ground truth based on the true parameter vector.
        y(l) = activation(dot(X_test, true_theta(l,:)));
        % Calculate the prediction based on the inferred parameter vector.
        prediction(l) = activation(dot(X_test, theta(l,:)));
    end
    % Calculate the error of the prediction.
    err = immse(y,prediction);
    fprintf('%s %d\n', 'Prediction error:', err);
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

% L1 norm of vector x of dimension n.
function y = lasso(x, n, lambda)
    y = x;
    penalty = norm(y,1);
    for i = 1:n
        if y(i) < 0
            y(i) = y(i) + lambda * penalty;
        elseif y(i) > 0
            y(i) = y(i) - lambda * penalty;
        end
    end
end

% Identity function.
function y = identity(x)
    y = x;
end

% Mean squared error.
function err = MSE(series, theta, N, L, d)
    err = 0;
    for s = d+1:N
        X = series((s-d):(s-1),:);
        X = reshape(X.',1,[]);
        for l = 1:L
            y = series(s,l);
            y_pred = dot(X, theta(l,:));
            err = err + mean(sumsqr(y - y_pred));
        end
    end
    err = err/N;
end

% Log-it activation function.
function y = sigmoid(x)
    y = 1/(1+exp(-x));
end

% Logistic loss gradient.
function grad = log_loss_gradient(X, y, y_pred, l, L, d)
    % Gradient of the parameter vector w.r.t. log-loss.
    grad = X * (y_pred(l) - y(l)) / (L*d);
end

% Logistic loss.
function err = log_loss(series, theta, N, L, d)
    err = 0;
    for s = d+1:N
        X = series((s-d):(s-1),:);
        X = reshape(X.',1,[]);
        for l = 1:L
            y = series(s,l);
            y_pred = sigmoid(dot(X, theta(l,:)));
            mse = mean(-y * log(y_pred) - (1 - y) * log(1 - y_pred));
            if isnan(mse) || isinf(mse)
                mse = 0;
            end
            err = err + mse;
        end
    end
    err = err/N;
end

% Gradient descent for time series of 2-D Bernouilli events.
function [theta, i, error] = estimate_parameters(series, N, L, d, rate, max_error, max_iterations, activation, loss, gradient, reg, lambda)
    %{
    param series : a time series of 2-D categorical events
    param N : lenght of the time series
    param L : number of locations in the 2-D grid where categorical events take place
    param d : memory depth describing the depth of the autoregression
    param rate : learning rate for gradient descent
    param max_error : maximum tolerated error
    param max_iterations : maximum tolerated number of gradient steps
    %}
    theta = normrnd(0,1,L,d*L);
    y_pred = normrnd(0,1,1,L);
    error = [];
    error = [error loss(series, theta, N, L, d)];
    i = 1;

    while (i < max_iterations) && (error(i) >= max_error)
        % For each time instance in the time horizon from d to N.
        for s = d+1:N
            % Take values from the last d time instances.
            X = series((s-d):(s-1),:);
            X = reshape(X.',1,[]);
            y = series(s,:);
            % For each location in the 2D grid of the current time instance.
            for l = 1:L
                % Predict the value of the current time instance.
                y_pred(l) = activation(dot(X, theta(l,:)));
                % Update the parameter vector.
                theta_grad = gradient(X, y, y_pred, l, L, d);
                theta(l,:) = theta(l,:) - rate * theta_grad;
                if exist('reg','var') && exist('lambda','var')
                    theta(l,:) = reg(theta(l,:), d*L, lambda);
                end
            end
        end
        % Calculate the prediction error over the time horizon.
        error = [error loss(series, theta, N, L, d)];
        i = i+1;
    end
end

function [time_horizon, N, L, true_theta] = generate_series(L_rows, L_columns, d, periods, density)
    L = L_rows*L_columns;
    % Creating random events with some density over an n by m grid.
    N = d + d*periods;
    % Initialiazing the time horizon.
    time_horizon = zeros(N,L);
    % Create a random Bernoulli process grid at the initial time strech.
    for s = (1:d)
        x = sprand(L, 1, density);
        x(x>0) = 1;
        for l = 1:L
            time_horizon(s,:) = x;
        end
    end
    % Initialising the true parameter vector and the bias.
    true_theta = normrnd(0, 1, L, d*L);
    % Generate time series.
    for s = (d+1):N
        % Predictor X of dimension d*L.
        X = time_horizon((s-d):(s-1),:);
        X = reshape(X.',1,[]);
        for l = 1:L
            time_horizon(s,l) = binary_sigmoid(dot(X, true_theta(l,:)));
        end
    end
end
