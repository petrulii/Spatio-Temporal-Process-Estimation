rows = 5;
cols = 5;
A = zeros(rows, cols);
B = ones(rows, cols);
B(2,3) = 0.2;
disp(A+B*0.5+2);