function [] = result_plot(x_values, zer_log_l1, error_log_l1, theta_norm_log_l1, theta_mse_log_l1, zer_lin_l1, error_lin_l1, theta_norm_lin_l1, theta_mse_lin_l1)
    f1 = figure('visible','on');
    hold on;
    plot(x_values, zer_log_l1);
    plot(x_values, zer_lin_l1);
    title('Non-zeros in the estimated parameter vector \beta');
    %xlabel('Memeory depth d');
    %xlabel('Length of the series N');
    xlabel('Lasso penalty hyper-parameter \lambda');
    ylabel('Non-zero values in \beta');
    legend('Maximum likelihood + L1','Least squares + L1');
    saveas(f1, 'zeros', 'fig');
    hold off;
    f2 = figure('visible','on');
    hold on;
    plot(x_values, error_log_l1);
    plot(x_values, error_lin_l1);
    title('Prediction error of the response vector at t+1');
    xlabel('Lasso penalty hyper-parameter \lambda');
    ylabel('MSE of Y_{N+1}^{*} and Y_{N+1}');
    legend('Maximum likelihood + L1','Least squares + L1');
    saveas(f2, 'MSE_prediction', 'fig');
    hold off;
    f3 = figure('visible','on');
    hold on;
    plot(x_values, theta_norm_log_l1);
    plot(x_values, theta_norm_lin_l1);
    title('Prediction error of the true par. vector');
    xlabel('Lasso penalty hyper-parameter \lambda');
    ylabel('2-norm of \beta^* - \beta');
    legend('Maximum likelihood + L1','Least squares + L1');
    saveas(f3, '2norm_theta', 'fig');
    hold off;
    f4 = figure('visible','on');
    hold on;
    plot(x_values, theta_mse_log_l1);
    plot(x_values, theta_mse_lin_l1);
    title('MSE of the estimated parameter vector');
    xlabel('Lasso penalty hyper-parameter \lambda');
    ylabel('MSE of \beta^* and \beta');
    legend('Maximum likelihood + L1','Least squares + L1', 'Least squares + L1 + const.');
    saveas(f4, 'MSE_theta', 'fig');
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