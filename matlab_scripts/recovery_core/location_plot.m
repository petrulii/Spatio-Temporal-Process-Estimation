function [] = location_plot(L, N, d, theta_true, theta0_true, theta, theta0, loc, init_grids)
    % For plotting.
    N = N+int32(N/2);
    norm_X_true = zeros(1,N);
    norm_probability_true = zeros(1,N);
    norm_X = zeros(1,N);
    norm_probability = zeros(1,N);
    % Initialiazing the time horizon.
    time_horizon = zeros(N,L);
    time_horizon_true = zeros(N,L);
    for s = (1:d)
        for l = 1:L
            time_horizon(s,:) = init_grids(s,:);
            time_horizon_true(s,:) = init_grids(s,:);
        end
    end
    % Generate time series.
    for s = (d+1):(N)
        % Predictor X of dimension d*L.
        X = time_horizon((s-d):(s-1),:);
        X = reshape(X.',1,[]);
        X_true = time_horizon_true((s-d):(s-1),:);
        X_true = reshape(X_true.',1,[]);
        for l = 1:L
            p = sigmoid(theta0(l) + dot(X, theta(l,:)));
            time_horizon(s,l) = Bernouilli_draw(p);
            p_true = sigmoid(theta0_true(l) + dot(X_true, theta_true(l,:)));
            time_horizon_true(s,l) = Bernouilli_draw(p_true);
            % Debugging
            disp('p_true:');
            disp(sigmoid(theta0_true(l) + dot(X_true, theta_true(l,:))));
            disp('p:');
            disp(sigmoid(theta0(l) + dot(X, theta(l,:))));
            % End of debug
            % Calculating the probabilities for one location.
            if l == loc
                norm_probability(1,s) = p;
                norm_X(1,s) = time_horizon(s,l);
                norm_probability_true(1,s) = p_true;
                norm_X_true(1,s) = time_horizon_true(s,l);
            end
        end
    end
    % Plotting
    time = 1:N;
    base_proba = sigmoid(theta0(loc));
    norm_theta0 = repelem(base_proba, N);
    base_proba_true = sigmoid(theta0_true(loc));
    norm_theta0_true = repelem(base_proba_true, N);
    f0 = figure('visible','on');
    hold on;
    plot(time, norm_theta0);
    plot(time, norm_probability);
    scatter(time, norm_X);
    plot(time, norm_theta0_true);
    plot(time, norm_probability_true);
    scatter(time, norm_X_true);
    xlabel('Time t');
    ylabel('Values at one location');
    title('True and estimated time series at one location');
    legend('\beta_0','\beta','y_{l, t}', '\beta_0^*','\beta^*','y_{l, t}^*');
    saveas(f0, '/home/im2ag/Desktop/M1/Internship/ljk-dao-internship/matlab_scripts/figures/predicted_true_series', 'png');
    hold off;
end