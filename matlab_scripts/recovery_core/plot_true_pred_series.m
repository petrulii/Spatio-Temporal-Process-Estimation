function [] = plot_true_pred_series(L, N_true, d, theta_true, theta0_true, theta, theta0)
    % For plotting.
    N = N_true+round(N_true);
    norm_probability_true = zeros(1,N);
    norm_probability_pred = zeros(1,N);
    % Initialiazing the time horizon.
    time_horizon = zeros(N,L);
    probabilities_true = zeros(N,L);
    probabilities_pred = zeros(N,L);
    time_horizon_pred = zeros(N,L);
    for s = (1:d)
        x = normrnd(0, 1, 1, L);
        x(x>=0) = 1;
        x(x<0) = 0;
        for l = 1:L
            time_horizon(s,:) = x;
            time_horizon_pred(s,:) = x;
        end
    end
    % Generate time series.
    for s = (d+1):(N)
        X = time_horizon((s-d):(s-1),:);
        X_pred = time_horizon_pred((s-d):(s-1),:);
        X = reshape(X.',1,[]);
        X_pred = reshape(X_pred.',1,[]);
        for l = 1:L
            p_true = sigmoid(theta0_true(l) + dot(X, theta_true(l,:)));
            p_pred = sigmoid(theta0(l) + dot(X_pred, theta(l,:)));
            time_horizon(s,l) = Bernouilli_draw(p_true);
            time_horizon_pred(s,l) = Bernouilli_draw(p_pred);
            probabilities_true(s,l) = p_true;
            probabilities_pred(s,l) = p_pred;
            % Calculating the 2-norms.
            norm_probability_true(1,s) = norm_probability_true(1,s) + p_true^2;
            norm_probability_pred(1,s) = norm_probability_pred(1,s) + p_pred^2;
        end
    end
    % Plotting
    time = 1:N;
    base_proba_true = sigmoid(theta0_true);
    norm_theta0_true = repelem(sqrt(sum(base_proba_true.^2)), N);
    norm_probability_true = sqrt(norm_probability_true);
    base_proba_pred = sigmoid(theta0);
    norm_theta0_pred = repelem(sqrt(sum(base_proba_pred.^2)), N);
    norm_probability_pred = sqrt(norm_probability_pred);
    f0 = figure('visible','on');
    hold on;
    plot(time, norm_theta0_pred, 'b');
    plot(time, norm_probability_pred, 'b');
    plot(time, norm_theta0_true, 'r');
    plot(time, norm_probability_true, 'r');
    % Draw A Red Vertical Line At ‘x=N’
    plot([1 1]*N_true, ylim, '-r');
    xlabel('Time t');
    ylabel('2-norm');
    title('True and estimated time series');
    legend('estimated P_0 at time t','estimated P at time t','true P_0^* at time t','true P^* at time t','N');
    saveas(f0, '/home/im2ag/Desktop/M1/Internship/ljk-dao-internship/matlab_scripts/figures/predicted_true_series', 'png');
    hold off;
end