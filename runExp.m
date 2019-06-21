%% running proxGD and acProxGD methods

m = 1000; % dim of features
n = 500; % # of samples
gamma = 0.75;

A = rand(m,n);
b = rand(m,1);

proxGD(A,b,gamma);