function x = ball_uniform(d)
    %alpha = normrnd(0,1,1,n);
    %alpha_norm = norm(alpha,2);
    %v = alpha/alpha_norm;
    % s is (d x n), n points in unit d-ball
    s = normrnd(0,1,1,d);
    norm = sqrt(sum(s.^2));
    r = rand().^(1/d);
    x = r*s/norm;
    %c = r./sqrt(sum(s.^2,1));
    %s = bsxfun(@times, s, c);
    %s = reshape(s.',1,[]);
end