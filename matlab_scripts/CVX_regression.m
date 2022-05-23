% Main function.
function main
    % Set the random seed.
    rng('default')
    % Set the solver
    clear all
    cvx_solver sedumi
    cvx_expert true
    % Set the parameters of the 2-D binary time series.
    rows = 10;
    columns = 10;
    % Memory depth.
    d = 3;
    % Number of locations.
    L = rows*columns;
    periods = 4;
    density = 0.01;
    
    % Generating Bernouilli time series of N time instances and L locations.
    [time_horizon, N, L, true_theta] = generate_series(rows, columns, d, periods, density);

    % Inferring theta, the parameter vector of the time series.
    cvx_begin
        variable theta(L, d*L);
        %variables x(2);
        obj = 0;
        for s = d:N
            X = time_horizon((s-d+1):s,:);
            X = reshape(X.',1,[]);
            %X = X(:);
            y = time_horizon(s+1,:);
            for l = 1:L
                theta_l = theta(l,:);
                obj = obj + (y(l)'*dot(X,theta_l)-sum(log_sum_exp([zeros(1,1); dot(X',theta_l')])));
            end
        end
        maximize(obj);
    cvx_end
    
    % Setting parameters for prediction.
    X_test = time_horizon((N-d)+1:N,:);
    X_test = reshape(X_test.',1,[]);
    % Generate a prediction and compare with groud truth.
    predict(X_test, L, true_theta, theta, rows, columns, @sigmoid, true);
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
    dist = norm((true_theta-theta),2);
    fprintf('%s %d\n', 'Prediction error:', err, 'distance between estimation and true parameters:', dist);
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
    for s = (d+1):(N+1)
        % Predictor X of dimension d*L.
        X = time_horizon((s-d):(s-1),:);
        X = reshape(X.',1,[]);
        for l = 1:L
            time_horizon(s,l) = binary_sigmoid(dot(X, true_theta(l,:)));
        end
    end
end