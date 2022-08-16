function [] = plot_series_one_location(L, N, d, theta, theta0, loc)
    % For plotting.
    norm_y = zeros(1,N);
    norm_probability = zeros(1,N);
    % Initialiazing the time horizon.
    time_series = zeros(N,L);
    for s = (1:d)
        x = normrnd(0, 1, 1, L);
        x(x>=0) = 1;
        x(x<0) = 0;
        for l = 1:L
            time_series(s,:) = x;
        end
    end
    % Generate time series.
    for s = (d+1):(N)
        % Predictor X of dimension d*L.
        X = time_series((s-d):(s-1),:);
        X = reshape(X.',1,[]);
        for l = 1:L
            p = sigmoid(theta0(l) + dot(X, theta(l,:)));
            time_series(s,l) = Bernouilli_draw(p);
            % Calculating the probabilities for one location.
            if l == loc
                norm_probability(1,s) = p;
                norm_y(1,s) = time_series(s,l);
            end
        end
    end
    % Plotting
    time = 1:N;
    base_proba = sigmoid(theta0(loc));
    norm_theta0 = repelem(base_proba, N);
    f0 = figure('visible','on');
    hold on;
    plot(time, norm_theta0);
    plot(time, norm_probability);
    scatter(time, norm_y);
    xlabel('Time t');
    ylabel('Values at one location');
    title('Time series at one location');
    legend('p0_{1,t}^*','p_{1,t}^*','y_{1,t}^*');
    saveas(f0, '/home/im2ag/Desktop/M1/Internship/ljk-dao-internship/matlab_scripts/figures/generated_series_one_location', 'png');
    hold off;
end