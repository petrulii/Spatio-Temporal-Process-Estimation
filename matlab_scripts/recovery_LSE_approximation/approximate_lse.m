function [y_max] = approximate_lse(r, approx, x)
    y_max = -1000;
    for k = 1:r
        a_k = approx(k,1);
        b_k = approx(k,2);
        y = a_k*x+b_k;
        if y>y_max
            y_max = y;
        end
    end
end