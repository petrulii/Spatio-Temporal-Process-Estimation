function [] = result_plot(x_values, zer_log_l1, error_log_l1, theta_norm_log_l1, theta_mse_log_l1, zer_lin_l1, error_lin_l1, theta_norm_lin_l1, theta_mse_lin_l1, zer_clin_l1, error_clin_l1, theta_norm_clin_l1, theta_mse_clin_l1)
    f1 = figure('visible','on');
    hold on;
    plot(x_values, zer_log_l1);
    plot(x_values, zer_lin_l1);
    plot(x_values, zer_clin_l1);
    title('Non-zeros in the estimated parameter vector \beta');
    xlabel('Log-scale lasso penalty hyper-parameter \lambda');
    %xlabel('Memeory depth d');
    %xlabel('Length of the series N');
    ylabel('Non-zero values in \beta');
    legend('Maximum likelihood + L1','Least squares + L1', 'Least squares + L1 + const.');
    saveas(f1, 'null_values_theta', 'png');
    hold off;
    f2 = figure('visible','on');
    hold on;
    plot(x_values, log(error_log_l1));
    plot(x_values, log(error_lin_l1));
    plot(x_values, log(error_clin_l1));
    title('Prediction error of the response vector at t+1');
    xlabel('Log-scale lasso penalty hyper-parameter \lambda');
    ylabel('Log-scale MSE of Y_{t+1}^{*} and Y_{t+1}');
    legend('Maximum likelihood + L1','Least squares + L1', 'Least squares + L1 + const.');
    saveas(f2, 'error_prediction', 'png');
    hold off;
    f3 = figure('visible','on');
    hold on;
    plot(x_values, theta_norm_log_l1);
    plot(x_values, theta_norm_lin_l1);
    plot(x_values, theta_norm_clin_l1);
    title('Prediction error of the true par. vector');
    xlabel('Log-scale lasso penalty hyper-parameter \lambda');
    ylabel('2-norm of \beta^* - \beta');
    legend('Maximum likelihood + L1','Least squares + L1', 'Least squares + L1 + const.');
    saveas(f3, 'distance_theta', 'png');
    hold off;
    f4 = figure('visible','on');
    hold on;
    plot(x_values, log(theta_mse_log_l1));
    plot(x_values, log(theta_mse_lin_l1));
    plot(x_values, log(theta_mse_clin_l1));
    title('MSE of the estimated parameter vector');
    xlabel('Log-scale lasso penalty hyper-parameter \lambda');
    ylabel('Log-scale MSE of \beta^* and \beta');
    legend('Maximum likelihood + L1','Least squares + L1', 'Least squares + L1 + const.');
    saveas(f4, 'mse_theta', 'png');
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
    xlabel(sp1, 'true \beta^*');
    shading interp;
    % This sets the limits of the colorbar to manual for the first plot
    caxis manual;
    caxis([bottom top]);
    colorbar;
    % Plotting the second plot
    sp2 = subplot(1,2,2);
    colormap('hot');
    imagesc(v_pred);
    xlabel(sp2, 'estimated \beta');
    shading interp;
    % This sets the limits of the colorbar to manual for the second plot
    caxis manual;
    caxis([bottom top]);
    colorbar;
    saveas(f, 'color_maps', 'png');
end