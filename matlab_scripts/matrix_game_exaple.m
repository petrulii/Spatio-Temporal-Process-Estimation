% P is the pay-off matrix, if player 1 (p1) chooses to play row i and
% player 2 (p2) chooses to play column j, then the p1 "pays" P_ij to p2.
% Here we consider a mixed-strategy game where u denotes the probability
% vector of p1 (resp. v for p2) where each k-th value of the vector u
% (resp. v) is the probability that p1 (resp. p2) will choose the k-th
% out of all possible p1 (resp. p2) strategies (u,v>=0 and sum(u)=sum(v)=1).
% Since each player makes his choice randomly and independently of the
% other player's choice according to the probability distribution described
% by the vectors u and v, the expected amount p1 pays is u'Pv and the
% expected gain for p2 is u'Pv. Therefore p1 wishes to choose u to minimize
% u'Pv, while p2 wishes to choose v to maximize u'Pv.

% Set the solver
clear all
cvx_solver sedumi

% Input data
rng(0);           % Setting the random seed to 0
n = 2;         % Number of rows
m = 2;         % Number of columns
P = [0.5488135  0.71518937; 0.60276338 0.54488318];%randn(n,m);   % Random pay-off matrix

% Optimal strategy for Player 1
cvx_begin
    variable u(n)
    minimize (max(P'*u))
    u >= 0;
    ones(1,n)*u == 1;
cvx_end

fprintf(1, "Best strategy vactor for player 1:\n");
disp(u);

% Optimal strategy for Player 2
cvx_begin
    variable v(m)
    maximize (min(P*v))
    v >= 0;
    ones(1,m)*v == 1;
cvx_end

fprintf(1, "Best strategy vector for player 2:\n")
disp(v);
