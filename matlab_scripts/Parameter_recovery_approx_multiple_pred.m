% Main function.
function Parameter_recovery_approx_multiple_pred
    % Set the random seed.
    rng(0);
    %cvx_solver mosek;
    % Dimensions of 2-D space grid.
    row = 4;
    col = row;
    % Memeory depth.
    d = 3;
    % Values used in parameter generation.
    radius = 1;
    values = [-1 -1];
    % Linear approximation of log-sum-exp.
    r = 10;
    [A_apprx, b_apprx] = battlse(r);
    % Multiple-step prediction parameter.
    pred_steps = 2;

    periods = 30;
    % Generating Bernouilli time series of N+1 time instances and L locations.
    [time_series, probabilities, N, L, true_theta, true_theta0] = generate_series(row, col, d, periods, 'operator', radius, values);

    lbd = 0.0005;
    % Maximum likelihood estimation with lasso.
    [theta_ml, theta0_ml] = logistic(time_series, N, L, d, lbd, A_apprx, b_apprx);
    
    lbd = 0.01;
    % Least squares estimation with lasso.
    [theta_ls, theta0_ls] = linear(time_series, N, L, d, lbd);
    % Generate a prediction and compare with groud truth.
    predict(time_series, N, L, d, true_theta, true_theta0, theta_ml, theta0_ml, theta_ls, theta0_ls, pred_steps, row, col);
    series_plot(L, N, d, true_theta, true_theta0, theta_ml, theta0_ml);
end

% Maximum likelihood estimation.
function [theta, init_intens] = logistic(time_series, N, L, d, lambda, A_apprx, b_apprx)
        cvx_begin;
            variable theta(L, d*L);
            variable init_intens(1, L);
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
                    obj = obj + (y*(dot(X,a)+b) - max(A_apprx*[0 (dot(X,a)+b)].'+b_apprx));
                end
            end
            obj = obj/((N-d)*L) - lambda * (sum(sum(abs(theta))) + sum(abs(init_intens)));
            maximize(obj);
        cvx_end;
        % Transform small values of theta to 0s.
        theta(theta>-0.0001 & theta<0.0001) = 0;
end

% Least-squares estimation.
function [theta, init_intens] = linear(time_series, N, L, d, lambda)
        cvx_begin
            variable theta(L, d*L);
            variable init_intens(1, L);
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
            obj = obj/((N-d)*L) + lambda * (sum(sum(abs(theta))) + sum(abs(init_intens)));
            minimize(obj);
        cvx_end
        % Transform small values of theta to 0s.
        theta(theta>-0.0001 & theta<0.0001) = 0;
end

% Prediction for time series of 2-D Bernouilli events.
function [err, zer, dist] = predict(time_series, N, L, d, true_theta, true_theta0, theta_ml, theta0_ml, theta_ls, theta0_ls, pred_steps, row, col)
    y = zeros(pred_steps,L);
    prediction_ml = zeros(pred_steps,L);
    prediction_ls = zeros(pred_steps,L);
    % For each location in the 2-D grid generate the true series and the prediction.
    for s = (N+1):(N+pred_steps)
        % Predictor X of dimension d*L.
        X = time_series((s-d):(s-1),:);
        X = reshape(X.',1,[]);
        for l = 1:L
            y(s,l) = sigmoid(true_theta0(l) + dot(X, true_theta(l,:)));
            prediction_ml(s,l) = sigmoid(theta0_ml(l) + dot(X, theta_ml(l,:)));
            prediction_ls(s,l) = identity(theta0_ls(l) + dot(X, theta_ls(l,:)));
        end
    end
    %compare_predictions(pred_steps, y, prediction_ml, prediction_ls, row, col);
end

function [] = compare_predictions(pred_steps, y, prediction_ml, prediction_ls, row, col)
    figure('visible','on');
    hold on;
    % Plotting
    time = 1:pred_steps;
    plot(time, total_variation(y, pred_steps, row, col));
    plot(time, total_variation(prediction_ml, pred_steps, row, col));
    plot(time, total_variation(prediction_ls, pred_steps, row, col));
    xlabel('Time t');
    ylabel('Total variation');
    title('Total variation of the time series');
    %legend('V(\nabla Y_t)');
    hold off;
end

function V = total_variation(series, pred_steps, row, col)
    V = zeros(1,pred_steps);
    for t=1:pred_steps
        grid = reshape(series(t,:),row,col)';
        [Gmag, Gdir] = imgradient(grid,'intermediate');
        V(t) = (sum(abs(Gmag(:))));
    end
end

% Identity activation function.
function y = identity(x)
    y = x;
end
