% Main function.
function Mosek_CVX_regression
    % Set the random seed.
    rng('default');
    % The length of the time horizon is d*periods+1.
    periods = 7;
    % Dimensions of 2-D space at any time instance.
    row = 7;
    col = row;
    % Norm for regularization (1-lasso, 2-ridge).
    reg = 1;
    % Lists for plotting
    error_log_l1 = [];
    error_lin_l1 = [];
    zer_log_l1 = [];
    zer_lin_l1 = [];
    theta_norm_log_l1 = [];
    theta_norm_lin_l1 = [];
    error_log_l2 = [];
    error_lin_l2 = [];
    zer_log_l2 = [];
    zer_lin_l2 = [];
    theta_norm_log_l2 = [];
    theta_norm_lin_l2 = [];
    error_log_projl1 = [];
    error_lin_projl1 = [];
    zer_log_projl1 = [];
    zer_lin_projl1 = [];
    theta_norm_log_projl1 = [];
    theta_norm_lin_projl1 = [];
    all_lambda = [];
    all_depths = [];
    
    % Memory depths.
    all_depths = linspace(2,7,6);
    % Regularization hyper-parameter.
    lbd = 0.001;
    
    for d = all_depths
        
        fprintf('%s %d', 'depth:',d);
        % Determine density.
        density = 0.1;
        % Generating Bernouilli time series of N time instances and L locations.
        [time_horizon, N, L, true_theta, true_b] = generate_series(row, col, d, periods, density);

        % LGR+LASSO : Logistic regression with lasso.
        [theta, b] = logistic(time_horizon, N, L, d, 1, lbd);
        % Generate a prediction and compare with groud truth.
        [err_log_l1, z_log_l1, t_n_log_l1] = predict(time_horizon((N-d)+1:N,:), time_horizon(N+1,:), L, d, true_theta, theta, b, row, col, @sigmoid, false);
        zer_log_l1 = [zer_log_l1 z_log_l1];
        error_log_l1 = [error_log_l1 err_log_l1];
        theta_norm_log_l1 = [theta_norm_log_l1 t_n_log_l1];
        
        % LNR+LASSO : Linear regression with lasso.
        [theta, b] = linear(time_horizon, N, L, d, 1, lbd);
        % Generate a prediction and compare with groud truth.
        [err_lin_l1, z_lin_l1, t_n_lin_l1] = predict(time_horizon((N-d)+1:N,:), time_horizon(N+1,:), L, d, true_theta, theta, b, row, col, @identity, false);
        zer_lin_l1 = [zer_lin_l1 z_lin_l1];
        error_lin_l1 = [error_lin_l1 err_lin_l1];
        theta_norm_lin_l1 = [theta_norm_lin_l1 t_n_lin_l1];
        
        % LGR+RIDGE : Logistic regression with ridge.
        [theta, b] = logistic(time_horizon, N, L, d, 2, lbd);
        % Generate a prediction and compare with groud truth.
        [err_log_l2, z_log_l2, t_n_log_l2] = predict(time_horizon((N-d)+1:N,:), time_horizon(N+1,:), L, d, true_theta, theta, b, row, col, @sigmoid, false);
        zer_log_l2 = [zer_log_l2 z_log_l2];
        error_log_l2 = [error_log_l2 err_log_l2];
        theta_norm_log_l2 = [theta_norm_log_l2 t_n_log_l2];
        
        % LNR+RIDGE : Linear regression with ridge.
        [theta, b] = linear(time_horizon, N, L, d, 2, lbd);
        % Generate a prediction and compare with groud truth.
        [err_lin_l2, z_lin_l2, t_n_lin_l2] = predict(time_horizon((N-d)+1:N,:), time_horizon(N+1,:), L, d, true_theta, theta, b, row, col, @identity, false);
        zer_lin_l2 = [zer_lin_l2 z_lin_l2];
        error_lin_l2 = [error_lin_l2 err_lin_l2];
        theta_norm_lin_l2 = [theta_norm_lin_l2 t_n_lin_l2];
        
        % LGR+PROJ_L1 : Logistic regression with projection onto L1 ball.
        [theta, b] = logistic_constraints(time_horizon, N, L, d);
        % Generate a prediction and compare with groud truth.
        [err_log_projl1, z_log_projl1, t_n_log_projl1] = predict(time_horizon((N-d)+1:N,:), time_horizon(N+1,:), L, d, true_theta, theta, b, row, col, @sigmoid, false);
        zer_log_projl1 = [zer_log_projl1 z_log_projl1];
        error_log_projl1 = [error_log_projl1 err_log_projl1];
        theta_norm_log_projl1 = [theta_norm_log_projl1 t_n_log_projl1];
        
        % LNR+PROJ_L1 : Linear regression with projection onto L1 ball.
        [theta, b] = linear_constraints(time_horizon, N, L, d);
        % Generate a prediction and compare with groud truth.
        [err_lin_projl1, z_lin_projl1, t_n_lin_projl1] = predict(time_horizon((N-d)+1:N,:), time_horizon(N+1,:), L, d, true_theta, theta, b, row, col, @identity, false);
        zer_lin_projl1 = [zer_lin_projl1 z_lin_projl1];
        error_lin_projl1 = [error_lin_projl1 err_lin_projl1];
        theta_norm_lin_projl1 = [theta_norm_lin_projl1 t_n_lin_projl1];
    end
    result_plot(all_depths, zer_log_l1, error_log_l1, theta_norm_log_l1, zer_log_l2, error_log_l2, theta_norm_log_l2, zer_lin_l1, error_lin_l1, theta_norm_lin_l1, zer_lin_l2, error_lin_l2, theta_norm_lin_l2, zer_log_projl1, error_log_projl1, theta_norm_log_projl1, zer_lin_projl1, error_lin_projl1, theta_norm_lin_projl1);
end

% Maximum likelihood estimation.
function [theta, init_intens] = logistic(time_horizon, N, L, d, reg, lambda)
        cvx_solver mosek;
        cvx_begin;
            variable theta(L, d*L);
            variable init_intens(L);
            obj = 0;
            for s = d:N
                X = time_horizon((s-d+1):s,:);
                X = reshape(X.',1,[]);
                % For each location in the 2-D grid.
                for l = 1:L
                    y = time_horizon(s+1,l);
                    a = theta(l,:);
                    b = init_intens(l);
                    % Log-likelihood with L1 penalty.
                    obj = obj + (y*(dot(X,a)+b) - log_sum_exp([0; (dot(X,a)+b)])) - lambda * norm(a,reg);
                end
            end
            maximize(obj);
        cvx_end;
        % Transform small values of theta to 0s.
        theta(theta>-0.001 & theta<0.001) = 0;
end

% Least-squares estimation.
function [theta, init_intens] = linear(time_horizon, N, L, d, reg, lambda)
        cvx_begin
            variable theta(L, d*L);
            variable init_intens(L);
            obj = 0;
            for s = d:N
                X = time_horizon((s-d+1):s,:);
                X = reshape(X.',1,[]);
                % For each location in the 2-D grid.
                for l = 1:L
                    y = time_horizon(s+1,l);
                    a = theta(l,:);
                    b = init_intens(l);
                    % Distance.
                    obj = obj + norm(y-(dot(X,a)+b))/2 + lambda * norm(a,reg);
                end
            end
            minimize(obj);
        cvx_end
        % Transform small values of theta to 0s.
        theta(theta>-0.001 & theta<0.001) = 0;
end

% Maximum likelihood estimation with constraints.
function [theta, init_intens] = logistic_constraints(time_horizon, N, L, d)
        cvx_solver mosek;
        cvx_begin;
            variable theta(L, d*L);
            variable init_intens(L);
            obj = 0;
            for s = d:N
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
            maximize(obj);
        cvx_end;
        for l = 1:L
            theta(l,:) = ProjectL1Ball(theta(l,:), 1, -1*ones(1, L*d), ones(1, L*d));
        end
        % Transform small values of theta to 0s.
        theta(theta>-0.001 & theta<0.001) = 0;
end

% Least-squares estimation with constraints.
function [theta, init_intens] = linear_constraints(time_horizon, N, L, d)
        cvx_begin
            variable theta(L, d*L);
            variable init_intens(L);
            obj = 0;
            for s = d:N
                X = time_horizon((s-d+1):s,:);
                X = reshape(X.',1,[]);
                % For each location in the 2-D grid.
                for l = 1:L
                    y = time_horizon(s+1,l);
                    a = theta(l,:);
                    b = init_intens(l);
                    % Distance.
                    obj = obj + norm(y-(dot(X,a)+b))/2;
                end
            end
            minimize(obj);
        cvx_end
        for l = 1:L
            theta(l,:) = ProjectL1Ball(theta(l,:), 1, -1*ones(1, L*d), ones(1, L*d));
        end
        % Transform small values of theta to 0s.
        theta(theta>-0.001 & theta<0.001) = 0;
end

% Prediction for time series of 2-D Bernouilli events.
function [err, zer, dist] = predict(X, y, L, d, true_theta, theta, b, rows, columns, activation, heatmap)
    if ~exist('heatmap','var')
        heatmap = false; end
    X = reshape(X.',1,[]);
    y = reshape(y.',1,[]);
    prediction = normrnd(0,1,1,L);
    obj = 0;
    for l = 1:L
        prediction(l) = activation(b(l) + dot(X, theta(l,:)));
        a = theta(l,:);
        obj = obj + (y(l)*(dot(X,a)+b(l)) - log(1+exp(dot(X,a)+(l))));
    end
    % Error of the prediction.
    err = immse(y,prediction);
    % 2-norm of the difference btween estimated theta and true theta.
    dist = sum(sqrt((true_theta-theta).^2),'all');
    zer = nnz(theta);
    % Null values in the estimated theta.
    zer = zer/(d*L*L);
    fprintf('%s\n', 'First values of estimated theta:');
    disp(theta(1,1:3));
    fprintf('%s\n', 'First values of true theta:');
    disp(true_theta(1,1:3));
    % Plot the true theta and predicted theta heatmaps.
    if heatmap == true
        true_th = reshape(true_theta(1,:),d,L);
        th = reshape(theta(1,:),d,L);
        color_plot(true_th, th);
    end
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
    % Creating random events with some density over an n by m grid.
    N = d + d*periods;
    % For plotting.
    norm_X = zeros(1,N+1);
    norm_lc = zeros(1,N+1);
    % Initialiazing the time horizon.
    time_horizon = zeros(N,L);
    % Create a random Bernoulli process grid at the initial time strech.
    for s = (1:d)
        x = sprand(L, 1, density);
        x(x>0) = 1;
        norm_X(s) = norm(x,2);
        for l = 1:L
            time_horizon(s,:) = x;
        end
    end
    % Initialising the sparse true parameter vector and the initial probability.
    %theta = normrnd(0, 1, L, d*L);
    theta = sprandn(L, d*L, density);
    fprintf('%s\n %d\n', 'Part of non-zero values in the true parameter vector:', nnz(theta)/(d*L*L));
    fprintf('%s\n', 'First values of the parameter vector:');
    disp(theta(1:2*d,1:2*d));
    % Putting half of the true parameter vector values below 0.
    theta0 = normrnd(0, 1, 1, L);
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
            % Calculating the norm of the linear combination.
            norm_lc(1,s) = norm_lc(1,s) + p;
            norm_X(1,s) = norm_X(1,s) + time_horizon(s,l);
        end
    end
    fprintf('%s\n %d\n', 'Part of non-zero values in the time horizon:', nnz(time_horizon)/(N*L));
    fprintf('%s\n', 'Last values of the time horizon:');
    disp(time_horizon(N-2*d:N,L-8:L));
    % Plotting
    time = 1:(N+1);
    norm_theta0 = repelem(norm(sigmoid(theta0),2), N+1);
    norm_theta = repelem(norm(sigmoid(theta),2), N+1);
    figure(1);
    hold on;
    plot(time, norm_theta0);
    plot(time, norm_theta);
    plot(time, norm_lc);
    plot(time, norm_X);
    xlabel('Time t');
    ylabel('2-norm');
    legend('sig(\theta_0)','sig(\theta)','sig(\theta^T X_t)','X_t');
    fprintf('%s %d \n', 'Norm of sig(theta):', norm(sigmoid(theta),2));
    saveas(gcf, 'time_horizon_vs_initial_intensities', 'png');
    hold off;
end

function [] = result_plot(all_lambda, zer_log_l1, error_log_l1, theta_norm_log_l1, zer_log_l2, error_log_l2, theta_norm_log_l2, zer_lin_l1, error_lin_l1, theta_norm_lin_l1, zer_lin_l2, error_lin_l2, theta_norm_lin_l2, zer_log_projl1, error_log_projl1, theta_norm_log_projl1, zer_lin_projl1, error_lin_projl1, theta_norm_lin_projl1)
    %all_lambda = log(all_lambda);
    f1 = figure('visible','off');
    hold on;
    plot(all_lambda, zer_log_l1);
    plot(all_lambda, zer_lin_l1);
    plot(all_lambda, zer_log_l2);
    plot(all_lambda, zer_lin_l2);
    plot(all_lambda, zer_lin_projl1);
    plot(all_lambda, zer_lin_projl1);
    %xlabel('Log-scale reguralization hyper-parameter \lambda');
    xlabel('Memeory depth d');
    ylabel('Non-zero values in \theta');
    legend('Log-regression + L1','Linear-regression + L1','Log-regression + L2','Linear-regression + L2','Log-regression + L1 ball proj.','Linear-regression + L1 ball proj.');
    saveas(f1, 'null_values', 'png');
    hold off;
    f2 = figure('visible','off');
    hold on;
    plot(all_lambda, log(error_log_l1));
    plot(all_lambda, log(error_lin_l1));
    plot(all_lambda, log(error_log_l2));
    plot(all_lambda, log(error_lin_l2));
    plot(all_lambda, log(error_log_projl1));
    plot(all_lambda, log(error_lin_projl1));
    xlabel('Memeory depth d');
    ylabel('Log-scale prediction error at t+1');
    legend('Log-regression + L1','Linear-regression + L1','Log-regression + L2','Linear-regression + L2','Log-regression + L1 ball proj.','Linear-regression + L1 ball proj.');
    saveas(f2, 'error', 'png');
    hold off;
    f3 = figure('visible','off');
    hold on;
    plot(all_lambda, log(theta_norm_log_l1));
    plot(all_lambda, log(theta_norm_lin_l1));
    plot(all_lambda, log(theta_norm_log_l2));
    plot(all_lambda, log(theta_norm_lin_l2));
    plot(all_lambda, log(theta_norm_log_projl1));
    plot(all_lambda, log(theta_norm_lin_projl1));
    xlabel('Memeory depth d');
    ylabel('Log-scale squared 2-norm of \theta - \theta^*');
    legend('Log-regression + L1','Linear-regression + L1','Log-regression + L2','Linear-regression + L2','Log-regression + L1 ball proj.','Linear-regression + L1 ball proj.');
    saveas(f3, 'distance', 'png');
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
    xlabel(sp1, 'true \theta');
    shading interp;
    % This sets the limits of the colorbar to manual for the first plot
    caxis manual;
    caxis([bottom top]);
    % Plotting the second plot
    sp2 = subplot(1,2,2);
    colormap('hot');
    imagesc(v_pred);
    xlabel(sp2, 'estimated \theta');
    shading interp;
    % This sets the limits of the colorbar to manual for the second plot
    caxis manual;
    caxis([bottom top]);
    colorbar;
    saveas(f, 'color_maps', 'png');
end