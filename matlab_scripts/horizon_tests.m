% Main function.
function horizon_tests
    % Set the random seed.
    rng(0);
    % Memory depth.
    d = 3;
    % The length of the time horizon is d*periods+1.
    periods = 9;
    % Dimensions of 2-D space at any time instance.
    row = 10;
    col = row;
    % Determine density.
    density = 0.1;
    % Generating Bernouilli time series of N time instances and L locations.
    [time_horizon, N, L, true_theta, theta0] = generate_series(row, col, d, periods, density);
    compare_density(L, N, d, true_theta, theta0, 3);
end

function [] = color_plot(v_true,v_pred,n)
    v_true = reshape(v_true,n,n);
    v_pred = reshape(v_pred,n,n);
    bottom = min(min(min(v_true)),min(min(v_pred)));
    top  = max(max(max(v_true)),max(max(v_pred)));
    f = figure('visible','off');
    % Plotting the first plot
    sp1 = subplot(1,2,1);
    colormap('hot');
    imagesc(v_true);
    xlabel(sp1, 'X at time t');
    shading interp;
    % This sets the limits of the colorbar to manual for the first plot
    caxis manual;
    caxis([bottom top]);
    % Plotting the second plot
    sp2 = subplot(1,2,2);
    colormap('hot');
    imagesc(v_pred);
    xlabel(sp2, 'X at time t+1');
    shading interp;
    % This sets the limits of the colorbar to manual for the second plot
    caxis manual;
    caxis([bottom top]);
    colorbar;
    saveas(f, 'color_maps', 'png');
end

function [] = compare_density(L, N, d, theta, theta0, loc)
    % For plotting.
    N = N+int32(N/2);
    norm_X = zeros(1,N);
    norm_probability = zeros(1,N);
    % Initialiazing the time horizon.
    time_horizon = zeros(N,L);
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
        for l = 1:L
            p = sigmoid(theta0(l) + dot(X, theta(l,:)));
            time_horizon(s,l) = Bernouilli_draw(p);
            % Calculating the probabilities for one location.
            if l == loc
                norm_probability(1,s) = p;
                norm_X(1,s) = time_horizon(s,l);
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
    scatter(time, norm_X);
    xlabel('Time t');
    ylabel('Values at one location');
    title('True and estimated time series at one location');
    legend('p0_{1,t}^*','p_{1,t}^*','y_{1,t}^*');
    saveas(f0, 'predicted_true_series', 'png');
    hold off;
end