function [] = Parameter_recovery_plot(x, zer_log_l1, error_log_l1, theta_norm_log_l1, zer_lin_l1, error_lin_l1, theta_norm_lin_l1, iterations)

    f1 = figure('visible','on');
    hold on;
    
    [zer_log_l1_Mean, zer_log_l1_CI95] = confidence_int(zer_log_l1);
    plot(x, zer_log_l1_Mean);
    CIs = zer_log_l1_CI95+zer_log_l1_Mean;
    display(CIs);
    display(CIs(0));
    display(CIs(1));
    ciplot(CIs(0), CIs(1), x);
    
    [zer_lin_l1_Mean, zer_lin_l1_CI95] = confidence_int(error_log_l1);
    plot(x, zer_lin_l1_Mean);
    CIs = zer_lin_l1_CI95+zer_lin_l1_Mean;
    ciplot(CIs(0), CIs(1), x);
    
    title('Non-zeros in the estimated parameter vector \beta');
    %xlabel('Memeory depth d');
    %xlabel('Length of the series N');
    xlabel('L1 penalty parameter \lambda');
    ylabel('Non-zero values in \beta');
    legend('Maximum likelihood + L1','Least squares + L1');
    saveas(f1, 'zeros', 'fig');
    hold off;
    
    zer_log_l1 = zer_log_l1/iterations;
    zer_lin_l1 = zer_lin_l1/iterations;
    theta_norm_log_l1 = theta_norm_log_l1/iterations;
    theta_norm_lin_l1 = theta_norm_lin_l1/iterations;
    
    f2 = figure('visible','on');
    hold on;
    [error_log_l1_Mean, error_log_l1_CI95] = confidence_int(error_log_l1);
    plot(x, error_log_l1_Mean);                                    % Plot Mean Of All Experiments
    CIs = error_log_l1_CI95+error_log_l1_Mean;
    ciplot(CIs(0), CIs(1), x)                                      % Plot 95% Confidence Intervals Of All Experiments
    [error_lin_l1_Mean, error_lin_l1_CI95] = confidence_int(error_lin_l1);
    plot(x, error_lin_l1_Mean);                                    % Plot Mean Of All Experiments
    plot(x, error_lin_l1_CI95+error_lin_l1_Mean);                  % Plot 95% Confidence Intervals Of All Experiments
    title('Prediction error of the probabilities at N+1');
    xlabel('L1 penalty parameter \lambda');
    ylabel('MSE of p_{N+1}^{*} and p_{N+1}');
    legend('Maximum likelihood + L1','Least squares + L1');
    saveas(f2, 'MSE_prediction', 'fig');
    hold off;
    
    f3 = figure('visible','on');
    hold on;
    [theta_norm_log_l1_Mean, theta_norm_log_l1_CI95] = confidence_int(error_log_l1);
    plot(x, theta_norm_log_l1_Mean);                                % Plot Mean Of All Experiments
    plot(x, theta_norm_log_l1_CI95+theta_norm_log_l1_Mean);         % Plot 95% Confidence Intervals Of All Experiments
    [theta_norm_lin_l1_Mean, theta_norm_lin_l1_CI95] = confidence_int(error_log_l1);
    plot(x, theta_norm_lin_l1_Mean);                                % Plot Mean Of All Experiments
    plot(x, theta_norm_lin_l1_CI95+theta_norm_lin_l1_Mean);         % Plot 95% Confidence Intervals Of All Experiments
    title('Error of the estimation of \beta^*');
    xlabel('L1 penalty parameter \lambda');
    ylabel('2-norm of \beta^* - \beta');
    legend('Maximum likelihood + L1','Least squares + L1');
    saveas(f3, 'theta_2norm', 'fig');
    hold off;
end

function [yMean, yCI95] = confidence_int(y)
    N = size(y,1);                                      % Number of ‘Experiments’ In Data Set
    yMean = mean(y);                                    % Mean Of All Experiments At Each Value Of ‘x’
    ySEM = std(y)/sqrt(N);                              % Compute ‘Standard Error Of The Mean’ Of All Experiments At Each Value Of ‘x’
    CI95 = tinv([0.025 0.975], N-1);                    % Calculate 95% Probability Intervals Of t-Distribution
    yCI95 = bsxfun(@times, ySEM, CI95(:));              % Calculate 95% Confidence Intervals Of All Experiments At Each Value Of ‘x’
    disp(yCI95);
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

function ciplot(lower,upper,x,colour)
     
% ciplot(lower,upper)       
% ciplot(lower,upper,x)
% ciplot(lower,upper,x,colour)
%
% Plots a shaded region on a graph between specified lower and upper confidence intervals (L and U).
% l and u must be vectors of the same length.
% Uses the 'fill' function, not 'area'. Therefore multiple shaded plots
% can be overlayed without a problem. Make them transparent for total visibility.
% x data can be specified, otherwise plots against index values.
% colour can be specified (eg 'k'). Defaults to blue.
% Raymond Reynolds 24/11/06
if length(lower)~=length(upper)
    error('lower and upper vectors must be same length')
end
if nargin<4
    colour='b';
end
if nargin<3
    x=1:length(lower);
end
% convert to row vectors so fliplr can work
if find(size(x)==(max(size(x))))<2
x=x'; end
if find(size(lower)==(max(size(lower))))<2
lower=lower'; end
if find(size(upper)==(max(size(upper))))<2
upper=upper'; end
fill([x fliplr(x)],[upper fliplr(lower)],colour)
end