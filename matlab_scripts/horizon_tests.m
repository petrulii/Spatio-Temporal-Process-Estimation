% Main function.
function horizon_tests
    % Set the random seed.
    rng(0);
    % Memory depth.
    d = 3;
    % The length of the time horizon is d*periods+1.
    periods = 50;
    % Dimensions of 2-D space at any time instance.
    row = 40;
    col = row;
    % Determine density.
    density = 0.1;
    % Generating Bernouilli time series of N time instances and L locations.
    [time_series, probabilities, N, L, true_theta, theta0] = generate_series(row, col, d, periods, density, 'disk');
    plot_one_location(L, N, d, true_theta, theta0, 3);
    color_plot_series(probabilities(1:15,:),15,row,col);
    color_plot_series(time_series(1:15,:),15,row,col);
end

function [] = color_plot_series(series,n,row,col)
    f = figure('visible','on');
    f.Position = [100 100 1400 90];
    % Plotting the first plot
    for t = 1:n
        sp = subplot(1,n,t);
        colormap(flipud(hot));
        grid = reshape(series(t,:),row,col)';
        imagesc(grid);
        xlabel(sp, t);
        shading interp;
    end
    saveas(f, 'series_heatmaps', 'png');
end

function [] = plot_one_location(L, N, d, theta, theta0, loc)
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
    saveas(f0, 'generated_series_one_location', 'png');
    hold off;
end