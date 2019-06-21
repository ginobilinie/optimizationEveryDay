%% This is a function for proximal gradient descent, and we use this example:
%% 1/2||Ax-b||^2+gamma*||x||_1
%% Algorithms:
%% z_k = x_k - t_k*A'(Ax_k-b)
%% x_k+1 = prox_h(z_k)
%% Dong Nie, June, 2019

function x = proxGD(A,b,gamma)
tic;
g = @(u) 0.5*norm(A*u-b)^2;
t = 1; %step size
beta = 0.5; % degeneration weight of step size
[m,n] = size(A);
x = zeros(n,1);
x_prev = x;
AtA = A'*A;
Atb = A'*b;
abs_tol = 1e-4;
rel_tol = 1e-2;
max_iter = 10000;%maximum training iterations

for k = 1: max_iter
    %s1: use g's grad to update z
    grad_x = AtA*x_prev - Atb;
    z = x - t*grad_x;
    %s2: x = prox(z) with soft_threshold
    x = soft_threshold(z, t*gamma)
    
    % s3: line search, if we donot use this step, fixed step size is used
    if g(x) <= g(x_prev) + grad_x'*(x - x_prev) + (1/(2*t))*(norm(x - x_prev))^2
        break;
    end
    t = beta*t;   %make the step smaller
    x_prev = x;
    
    h.prox_optval(k) = objective(A, b, gamma, x, x);
    if k > 1 && abs(h.prox_optval(k) - h.prox_optval(k-1)) < abs_tol
        break;
    end
end


% % obtain the optimal solution x and the corresponding function value: p_prox
h.x_prox = x; % the optimal solution
h.p_prox = h.prox_optval(end); % the corresponding function value

% % display information
h.prox_grad_toc = toc; % show running time
fprintf('Elapsed time for proxGD: %.3f seconds.\n', h.prox_grad_toc);
h.prox_iter = length(h.prox_optval);
K = h.prox_iter;
h.prox_optval = padarray(h.prox_optval', K-h.prox_iter, h.p_prox, 'post');

plot( 1:K, h.prox_optval, 'r+');
xlim([0 75]);

end 

% the objective function
function p = objective(A, b, gamma, x, z)
  p = 0.5*(norm(A*x - b))^2 + gamma*norm(z,1);
end

%soft threshold, which is the proximal operator for |x|
function x = soft_threshold(b,lambda)
  x = sign(b).*max(abs(b) - lambda,0);
end
