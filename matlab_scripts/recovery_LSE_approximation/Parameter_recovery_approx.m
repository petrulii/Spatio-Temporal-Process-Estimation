% Main function.
function Parameter_recovery_approx
    % Set the random seed.
    rng(0);
    %cvx_solver mosek;
    % The length of the time horizon is d*periods+1.
    all_periods = [2 10 20];
    len_periods = length(all_periods);
    %all_lambdas = logspace(-3,3,20);
    %len_lambdas = length(all_lambdas);
    % Dimensions of 2-D space grid.
    row = 4;
    col = row;
    % Memeory depth.
    d = 2;
    % Values used in parameter generation.
    radius = 1;
    values = [1 -1];
    % Lists for plotting.
    iterations = 2;
    all_N = zeros(1, len_periods);
    error_log_l1 = zeros(iterations,len_periods);
    error_lin_l1 = zeros(iterations,len_periods);
    zer_log_l1 = zeros(iterations,len_periods);
    zer_lin_l1 = zeros(iterations,len_periods);
    theta_norm_log_l1 = zeros(iterations,len_periods);
    theta_norm_lin_l1 = zeros(iterations,len_periods);
    % Linear approximation of log-sum-exp.
    r = 10;
    [A_apprx, b_apprx] = battlse(r);

    for i = 1:iterations
        % Regularization hyper-parameter.
        for j = 1:len_periods
            fprintf('\n\n\n%s %d %s %d\n\n', 'Iteration:', i, ', period index:', j);
            periods = all_periods(j);
            %lbd = all_lambdas(j);
            % Generating Bernouilli time series of N+1 time instances and L locations.
            [time_series, probabilities, N, L, true_theta, true_theta0] = generate_series(row, col, d, periods, 'operator', radius, values);
            all_N(j) = N;

            lbd = 0.0005;
            % Maximum likelihood estimation with lasso.
            [theta, theta0] = logistic(time_series, N, L, d, lbd, A_apprx, b_apprx);
            % Generate a prediction and compare with groud truth.
            [err_log_l1, z_log_l1, t_n_log_l1] = predict(time_series((N-d)+1:N,:), time_series(N+1,:), L, d, true_theta, theta, true_theta0, theta0, row, col, @sigmoid);
            zer_log_l1(i,j) = z_log_l1;
            error_log_l1(i,j) = err_log_l1;
            theta_norm_log_l1(i,j) = t_n_log_l1;

            lbd = 0.01;
            % Least squares estimation with lasso.
            [theta, theta0] = linear(time_series, N, L, d, lbd);
            % Generate a prediction and compare with groud truth.
            [err_lin_l1, z_lin_l1, t_n_lin_l1] = predict(time_series((N-d)+1:N,:), time_series(N+1,:), L, d, true_theta, theta, true_theta0, theta0, row, col, @identity);
            zer_lin_l1(i,j) = z_lin_l1;
            error_lin_l1(i,j) = err_lin_l1;
            theta_norm_lin_l1(i,j) = t_n_lin_l1;
            
            % Displaying calculated results.
            disp(all_N);
            disp(theta_norm_log_l1);
            disp(error_log_l1);
            disp(theta_norm_lin_l1);
            disp(error_lin_l1);
        end
    end
    Parameter_recovery_plot(all_N, zer_log_l1, error_log_l1, theta_norm_log_l1, zer_lin_l1, error_lin_l1, theta_norm_lin_l1, 'N');
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

% Identity activation function.
function y = identity(x)
    y = x;
end
