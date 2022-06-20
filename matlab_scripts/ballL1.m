function v = ballL1(n)
    alpha = normrnd(0,1,1,n);
    alpha_norm = norm(alpha,2);
    v = alpha/alpha_norm;
end