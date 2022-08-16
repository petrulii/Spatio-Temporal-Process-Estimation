% Main function.
function horizon_tests
    % Set the random seed.
    rng(0);
    % Memory depth.
    d = 3;
    % The length of the time horizon is d*periods+1.
    periods = 100;
    % Dimensions of 2-D space at any time instance.
    row = 3;
    col = row;
    % Define the value for an operator used in parameter generation.
    radius = 1;
    values = [1 1].*(-1);
    % Generating Bernouilli time series of N time instances and L locations.
    [time_series, probabilities, N, L, theta, theta0] = generate_series(row, col, d, periods, 'operator', radius, values);
    plot_series_one_location(L, N, d, theta, theta0, 1);
    plot_total_variation(N, probabilities, row, col);
    %plot_series(probabilities(4:124,:),120,row,col);
    %plot_series(time_series(4:124,:),120,row,col);
end

function [] = plot_total_variation(N, series, row, col)
    f = figure('visible','on');
    hold on;
    V = zeros(1,N);
    for t=1:N
        grid = reshape(series(t,:),row,col)';
        [Gmag, Gdir] = imgradient(grid,'intermediate');
        V(t) = (sum(abs(Gmag(:))));
    end
    % Plotting
    time = 1:N;
    plot(time, V);
    xlabel('Time t');
    ylabel('Total variation');
    title('Total variation of the time series');
    legend('V(\nabla Y_t)');
    saveas(f, 'total_variation', 'png');
    hold off;
end