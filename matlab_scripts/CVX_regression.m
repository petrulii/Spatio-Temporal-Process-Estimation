% Main function.
function main
    % Set the random seed.
    rng('default')
    % Set the solver
    %cvx_solver sedumi
    % Memory depth.
    d = 3;
    periods = 4;
    link = @sigmoid;
    error = [];
    distance = [];
    %dimension = [];
    all_lambda = [];
    row = 7;
    for lambda = 0:0.2:3
        col = row;
        density = (row+col)/(row*col);
        % Generating Bernouilli time series of N time instances and L locations.
        [time_horizon, N, L, true_theta] = generate_series(row, col, d, periods, density);
        % Inferring theta, the parameter vector of the time series.
        theta = estimate_parameters_lasso(time_horizon, N, L, d, lambda);
        % Generate a prediction and compare with groud truth.
        [err, dist] = predict(time_horizon((N-d)+1:N,:), time_horizon(N+1,:), L, true_theta, theta, row, col, link, false);
        error = [error err];
        distance = [distance dist];
        all_lambda = [all_lambda lambda];
        %dimension = [dimension row*col];
    end
    plot(all_lambda, distance);
    xlabel('Lambda');
    ylabel('Euclidean distance between \theta and \theta^*'); 
    saveas(gcf, 'distance', 'png');
    plot(all_lambda, error);
    xlabel('Lambda');
    ylabel('Prediction error at t+1'); 
    saveas(gcf, 'error', 'png');
end

function theta = estimate_parameters(time_horizon, N, L, d)
    %{
    param time_horizon : a time series of 2-D categorical events
    param N : lenght of the time series
    param L : number of locations in the 2-D grid where categorical events take place
    param d : memory depth describing the depth of the autoregression
    %}
    cvx_begin
        variable theta(L, d*L);
        obj = 0;
        for s = d:N
            X = time_horizon((s-d+1):s,:);
            X = reshape(X.',1,[]);
            y = time_horizon(s+1,:);
            for l = 1:L
                theta_l = theta(l,:);
                % Log-likelihood.
                obj = obj + (y(l)'*dot(X,theta_l)-sum(log_sum_exp([zeros(1,1); dot(X',theta_l')])));
            end
        end
        maximize(obj);
    cvx_end
end

function theta = estimate_parameters_lasso(time_horizon, N, L, d, lambda)
    %{
    param time_horizon : a time series of 2-D categorical events
    param N : lenght of the time series
    param L : number of locations in the 2-D grid where categorical events take place
    param d : memory depth describing the depth of the autoregression
    %}
    if lambda == 0
        theta = estimate_parameters(time_horizon, N, L, d);
    else
        cvx_begin
            variable theta(L, d*L);
            obj = 0;
            for s = d:N
                X = time_horizon((s-d+1):s,:);
                X = reshape(X.',1,[]);
                y = time_horizon(s+1,:);
                for l = 1:L
                    theta_l = theta(l,:);
                    % Log-likelihood with L1 penalty.
                    obj = obj + (y(l)'*dot(X,theta_l)-sum(log_sum_exp([zeros(1,1); dot(X',theta_l')]))) - lambda * norm(theta_l,1);
                end
            end
            maximize(obj);
        cvx_end
    end
end

% Prediction for time series of 2-D Bernouilli events.
function [err, dist] = predict(X, y, L, true_theta, theta, rows, columns, activation, heatmap)
    if ~exist('heatmap','var')
        heatmap = false; end
    X = reshape(X.',1,[]);
    y = reshape(y.',1,[]);
    prediction = normrnd(0,1,1,L);
    for l = 1:L
        prediction(l) = activation(dot(X, theta(l,:)));
    end
    % Calculate the error of the prediction.
    err = immse(y,prediction);
    dist = norm((true_theta-theta),2);
    %fprintf('%s %d\n', 'Prediction error:', err, 'distance between estimation and true parameters:', dist);
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
            % Train data.
            if s ~= (N+1)
                time_horizon(s,l) = binary_sigmoid(dot(X, true_theta(l,:)));
            % Test data.
            else
                time_horizon(s,l) = sigmoid(dot(X, true_theta(l,:)));
            end
        end
    end
end