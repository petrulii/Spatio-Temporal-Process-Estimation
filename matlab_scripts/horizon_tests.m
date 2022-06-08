% Main function.
function horizon_tests
    % Set the random seed.
    rng(0);
    % Memory depth.
    d = 3;
    % The length of the time horizon is d*periods+1.
    periods = 100;
    % Dimensions of 2-D space at any time instance.
    row = 10;
    col = row;
    % Determine density.
    density = 0.1;
    % Generating Bernouilli time series of N time instances and L locations.
    [time_horizon, N, L, true_theta, true_b, init_grids] = generate_series(row, col, d, periods, density);
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