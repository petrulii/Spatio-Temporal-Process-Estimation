function s = ball_uniform(d, n)
    %alpha = normrnd(0,1,1,n);
    %alpha_norm = norm(alpha,2);
    %v = alpha/alpha_norm;
    % s is (d x n), n points in unit d-ball
    s = randn(d,n);
    r = rand(1,n).^(1/d);
    c = r./sqrt(sum(s.^2,1));
    s = bsxfun(@times, s, c);
    s = reshape(s.',1,[]);
end