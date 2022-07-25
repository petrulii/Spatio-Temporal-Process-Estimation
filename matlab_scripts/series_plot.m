function [] = series_plot(L, N, d, theta_true, theta0_true, theta, theta0)
    % For plotting.
    N = N+(N/2);
    norm_X_true = zeros(1,N);
    norm_probability_true = zeros(1,N);
    norm_X = zeros(1,N);
    norm_probability = zeros(1,N);
    % Initialiazing the time horizon.
    time_horizon = zeros(N,L);
    time_horizon_true = zeros(N,L);
    for s = (1:d)
        x = normrnd(0, 1, 1, L);
        x(x>=0) = 1;
        x(x<0) = 0;
        for l = 1:L
            time_horizon(s,:) = x;
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
            % Calculating the 2-norms.
            norm_probability(1,s) = norm_probability(1,s) + p^2;
            norm_X(1,s) = norm_X(1,s) + time_horizon(s,l);
            norm_probability_true(1,s) = norm_probability_true(1,s) + p_true^2;
            norm_X_true(1,s) = norm_X_true(1,s) + time_horizon_true(s,l);
        end
    end
    % Plotting
    time = 1:N;
    base_proba = sigmoid(theta0);
    norm_theta0 = repelem(sqrt(sum(base_proba.^2)), N);
    norm_probability = sqrt(norm_probability);
    norm_X = sqrt(norm_X);
    base_proba_true = sigmoid(theta0_true);
    norm_theta0_true = repelem(sqrt(sum(base_proba_true.^2)), N);
    norm_probability_true = sqrt(norm_probability_true);
    norm_X_true = sqrt(norm_X_true);
    f0 = figure('visible','on');
    hold on;
    plot(time, norm_theta0);
    plot(time, norm_probability);
    scatter(time, norm_X);
    plot(time, norm_theta0_true);
    plot(time, norm_probability_true);
    scatter(time, norm_X_true);
    xlabel('Time t');
    ylabel('2-norm');
    title('True and estimated time series');
    legend('estimated \beta_0 at time t','estimated \beta at time t','predicted Y_t at time t', 'true \beta_0^* at time t','true \beta^* at time t','true Y_t^* at time t');
    saveas(f0, 'predicted_true_series', 'png');
    hold off;
end