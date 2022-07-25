function [] = Parameter_recovery_const_plot(x, zer_log_l1, zer_log_l1_const, error_log_l1, error_log_l1_const, theta_norm_log_l1, theta_norm_log_l1_const, x_label)

    % Non-zero value in the parameter vector plot.
    f1 = figure('visible','on');
    hold on;
    
    [zer_log_l1_Mean, zer_log_l1_CI95] = confidence_int(zer_log_l1);
    plot(x, zer_log_l1_Mean);
    CIs = zer_log_l1_CI95+zer_log_l1_Mean;
    patch([x, fliplr(x)], [CIs(1,:) fliplr(CIs(2,:))], 'b', 'EdgeColor','none', 'FaceAlpha',0.15)
    
    [zer_log_l1_const_Mean, zer_log_l1_const_CI95] = confidence_int(zer_log_l1_const);
    plot(x, zer_log_l1_const_Mean);
    CIs = zer_log_l1_const_CI95+zer_log_l1_const_Mean;
    patch([x, fliplr(x)], [CIs(1,:) fliplr(CIs(2,:))], 'y', 'EdgeColor','none', 'FaceAlpha',0.15)
    %{
    [zer_lin_l1_Mean, zer_lin_l1_CI95] = confidence_int(zer_lin_l1);
    plot(x, zer_lin_l1_Mean);
    CIs = zer_lin_l1_CI95+zer_lin_l1_Mean;
    patch([x, fliplr(x)], [CIs(1,:) fliplr(CIs(2,:))], 'y', 'EdgeColor','none', 'FaceAlpha',0.25)
    
    [zer_lin_l1_const_Mean, zer_lin_l1_const_CI95] = confidence_int(zer_lin_l1_const);
    plot(x, zer_lin_l1_const_Mean, 'r');
    CIs = zer_lin_l1_const_CI95+zer_lin_l1_const_Mean;
    patch([x, fliplr(x)], [CIs(1,:) fliplr(CIs(2,:))], 'r', 'EdgeColor','none', 'FaceAlpha',0.25)
    %}
    title('Non-zeros in the estimated parameter vector \beta');
    xlabel(x_label);
    ylabel('Non-zero values in \beta');
    legend('Maximum likelihood + L1','Maximum likelihood 95% CI','Maximum likelihood + L1 + const.','Maximum likelihood + const. 95% CI');
    saveas(f1, 'beta_non_zero_values', 'fig');
    hold off;
    
    % Prediction error plot.    
    f2 = figure('visible','on');
    hold on;

    [error_log_l1_Mean, error_log_l1_CI95] = confidence_int(error_log_l1);
    plot(x, error_log_l1_Mean);
    CIs = error_log_l1_CI95+error_log_l1_Mean;
    patch([x, fliplr(x)], [CIs(1,:) fliplr(CIs(2,:))], 'b', 'EdgeColor','none', 'FaceAlpha',0.15)

    [error_log_l1_const_Mean, error_log_l1_const_CI95] = confidence_int(error_log_l1_const);
    plot(x, error_log_l1_const_Mean);
    CIs = error_log_l1_const_CI95+error_log_l1_const_Mean;
    patch([x, fliplr(x)], [CIs(1,:) fliplr(CIs(2,:))], 'y', 'EdgeColor','none', 'FaceAlpha',0.15)
    %{
    [error_lin_l1_Mean, error_lin_l1_CI95] = confidence_int(error_lin_l1);
    plot(x, error_lin_l1_Mean);
    CIs = error_lin_l1_CI95+error_lin_l1_Mean;
    patch([x, fliplr(x)], [CIs(1,:) fliplr(CIs(2,:))], 'y', 'EdgeColor','none', 'FaceAlpha',0.25)
    
    [error_lin_l1_const_Mean, error_lin_l1_const_CI95] = confidence_int(error_lin_l1_const);
    plot(x, error_lin_l1_const_Mean, 'r');
    CIs = error_lin_l1_const_CI95+error_lin_l1_const_Mean;
    patch([x, fliplr(x)], [CIs(1,:) fliplr(CIs(2,:))], 'r', 'EdgeColor','none', 'FaceAlpha',0.25)
    %}
    title('Prediction error of the probabilities at N+1');
    xlabel(x_label);
    ylabel('MSE of p_{N+1}^{*} and p_{N+1}');
    legend('Maximum likelihood + L1','Maximum likelihood 95% CI','Maximum likelihood + L1 + const.','Maximum likelihood + const. 95% CI');
    saveas(f2, 'prediction_error_MSE', 'fig');
    hold off;
    
    % Estimation error plot.  
    f3 = figure('visible','on');
    hold on;
    
    [theta_norm_log_l1_Mean, theta_norm_log_l1_CI95] = confidence_int(theta_norm_log_l1);
    plot(x, theta_norm_log_l1_Mean);
    CIs = theta_norm_log_l1_CI95+theta_norm_log_l1_Mean;
    patch([x, fliplr(x)], [CIs(1,:) fliplr(CIs(2,:))], 'b', 'EdgeColor','none', 'FaceAlpha',0.15)
    
    [theta_norm_log_l1_const_Mean, theta_norm_log_l1_const_CI95] = confidence_int(theta_norm_log_l1_const);
    plot(x, theta_norm_log_l1_const_Mean);
    CIs = theta_norm_log_l1_const_CI95+theta_norm_log_l1_const_Mean;
    patch([x, fliplr(x)], [CIs(1,:) fliplr(CIs(2,:))], 'y', 'EdgeColor','none', 'FaceAlpha',0.15)
    %{
    [theta_norm_lin_l1_Mean, theta_norm_lin_l1_CI95] = confidence_int(theta_norm_lin_l1);
    plot(x, theta_norm_lin_l1_Mean);
    CIs = theta_norm_lin_l1_CI95+theta_norm_lin_l1_Mean;
    patch([x, fliplr(x)], [CIs(1,:) fliplr(CIs(2,:))], 'y', 'EdgeColor','none', 'FaceAlpha',0.25)
    
    [theta_norm_lin_l1_const_Mean, theta_norm_lin_l1_const_CI95] = confidence_int(theta_norm_lin_l1_const);
    plot(x, theta_norm_lin_l1_const_Mean, 'r');
    CIs = theta_norm_lin_l1_const_CI95+theta_norm_lin_l1_const_Mean;
    patch([x, fliplr(x)], [CIs(1,:) fliplr(CIs(2,:))], 'r', 'EdgeColor','none', 'FaceAlpha',0.25)
    %}
    title('Error of the estimation of \beta^*');
    xlabel(x_label);
    ylabel('2-norm of \beta^* - \beta');
    legend('Maximum likelihood + L1','Maximum likelihood 95% CI','Maximum likelihood + L1 + const.','Maximum likelihood + const. 95% CI');%,'Least squares + L1','Least squares 95% CI','Maximum likelihood + L1 + const.','Maximum likelihood + const. 95% CI','Least squares + L1 + const.','Least squares + const. 95% CI');
    saveas(f3, 'beta_estimation_error_2_norm', 'fig');
    hold off;
end

function [yMean, yCI95] = confidence_int(y)
    N = size(y,1);                                      % Number of ‘Experiments’ In Data Set
    yMean = mean(y);                                    % Mean Of All Experiments At Each Value Of ‘x’
    ySEM = std(y)/sqrt(N);                              % Compute ‘Standard Error Of The Mean’ Of All Experiments At Each Value Of ‘x’
    CI95 = tinv([0.025 0.975], N-1);                    % Calculate 95% Probability Intervals Of t-Distribution
    yCI95 = bsxfun(@times, ySEM, CI95(:));              % Calculate 95% Confidence Intervals Of All Experiments At Each Value Of ‘x’
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