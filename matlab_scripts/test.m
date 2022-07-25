A = [0 1 0 1 1, 1 1 1 0 0, 0 1 0 1 1, 0 0 0 1 1, 1 1 1 1 0];
B = [1 1 1, 1 1 1, 1 1 1].*(1);
C = conv2(A,B);
disp('2D convolution :');
disp(C(1:10));
D = sigmoid(C);
disp('Sigmoid :');
disp(D(1:10));
plot([1 2 3], [1 2 3]);
xline(2);