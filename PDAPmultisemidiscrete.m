function [u_opt, alg_out] = PDAPmultisemidiscrete(p, y_ref, alpha, phi, alg_opts)
%{
Solves the optimization problem
min (1/2)*|Sred*u - ref|^2 + alpha*phi(|u|_1,2)
Inputs
  p - a structure with auxilary functions and data types of the model  
  alpha - hyperparmeter
  phi - the penalizing function
  alg_opts - optional arguments
    .max_step - number of maximum step sizes
    .plot_every - plot at each this many iterations
    .sparsification - to remove nodes or not
    .TOL - tolerence
%}

N = p.N;
obj = p.obj;

%% setup options field
if (nargin <= 3)
    alg_opts = struct();
end

max_step = get_field_default(alg_opts, 'max_step', 1000);
TOL = get_field_default(alg_opts, 'TOL', 1e-5);
update_M0 = get_field_default(alg_opts, 'update_M0', true);
sparsification = get_field_default(alg_opts, 'sparsification', false);
optimize_x = get_field_default(alg_opts, 'optimize_x', false);

plot_final = get_field_default(alg_opts, 'plot_final', true);
plot_every = get_field_default(alg_opts, 'plot_every', 0);

% initial guess
u0 = get_field_default(alg_opts, 'u0', p.u_zero);
uk = u0;

% initial values
Ku = p.K(p, p.xhat, uk);
norms_u = computeNorm(uk.u, N);
j = obj.F(Ku - y_ref) + alpha*sum(phi.phi(norms_u));
suppsize = nnz(norms_u);

% algorithmic parameters: M0
M0 = min(phi.inv(obj.F(-y_ref)/alpha), 1e8);

% save algorithmic diagnostics
alg_out = struct();
alg_out.us{1} = uk;
alg_out.js = j;
alg_out.supps = suppsize;
alg_out.tics = 0;
alg_out.Psis = 0;

tic;

fprintf('PDAP: %3i, desc: (%1.0e,%1.0e), supp: %i, j: (%1.2e,%1.0e), M0: %1.2e\n', ...
    0, 0, 0, suppsize, j, Inf, M0);

iter = 1;
while true
    
    % compute maxima of gradient
    yk = obj.dF(Ku - y_ref);
    xinit = uk.x;
    if optimize_x
      xinit = p.u_zero.x;
    end
    xmax = p.find_max(p, yk, xinit);
    grad = reshape(p.Ks(p, xmax, p.xhat, yk), [p.N, size(xmax,2)]);
    
    %% Coordinate descent
    
    % dynamic update of the bounding parameter
    % improves convergence speed
    if update_M0
        M0 = min(phi.inv(j/alpha), 1e6);
    end
    
    % Compute maximum of gradient
    norms_grad = computeNorm(grad, N);

    [~, perm] = sort(norms_grad, 'descend');

    xmax = xmax(:,perm);
    grad = grad(:,perm);
    norms_grad = norms_grad(:,perm);
    
    %% insert multiple points
    % select promising points
    buffer = (1/2) * (max(norms_grad) - alpha);
    cutoff = alpha + buffer;
    locs = find(norms_grad >= min(alpha + buffer, norms_grad));

    % select maximum number of
    Npoint = 15;
    locs = locs(1:min(Npoint, length(locs)));
                
    xmax = xmax(:,locs);
    grad = grad(:,locs);
    norms_grad = norms_grad(:,locs);
    
    [max_grad, loc] = max(norms_grad);

    % compute potential descent direction
    coeff = - sqrt(eps) * grad ./ norms_grad;
    coeff(:,loc) = - grad(:,loc) / norms_grad(loc);
    newsupp = [uk.x, xmax];
    vhat = struct('x', newsupp, 'u', [zeros(size(uk.u)), coeff]);
    uk_new = struct('x', newsupp, 'u', [uk.u, 0*coeff]);

    % slope and curvature
    Kvhat = p.K(p, p.xhat, vhat);
    phat = real(Kvhat'*yk);
    what = real(Kvhat'*obj.ddF(Ku - y_ref)*Kvhat);
    
    % upper bound for the global functional error
    phi_u = alpha * sum(phi.phi(norms_u));
    phi_vhat = alpha * phi.phi(sum(M0 * computeNorm(vhat.u, N)));
    upperb = phi_u + real(Ku'*yk) - min(phi_vhat + M0*phat, 0);
    
    %if upperb < 0
    %    keyboard
    %end
    
    %termination crit.
    %if stepsize <= tol || (nnz(u) == 0 && nnz(newdirac) == 0)
    if (upperb <= TOL*j) || (iter > max_step)
        fprintf('PDAP: %3i, desc: (%1.0e,%1.0e), supp: %i, j: (%1.2e,%1.0e), M0: %1.2e\n', ...
            iter, 0, 0, suppsize, j, upperb, M0);

        alg_out.Psis(iter) = upperb;
        break;
    end

    %% coordinate descent step
    %  minimize: alpha * phi(tau) + phat * tau + (what/2) * tau^2
    %          = prox_{(alpha / what) * phi} (- phat / what)

    %tau = 1e-16;
    if phat <= -alpha
      tau = phi.prox(alpha/what, - phat/what);
    else
      tau = 0;
    end
    
    uk = uk_new;
    uk.u = uk.u + tau*vhat.u;
    
    % update values
    Ku = p.K(p, p.xhat, uk);
    norms_u = computeNorm(uk.u, N);
    newj1 = obj.F(Ku - y_ref) + alpha*sum(phi.phi(norms_u));
    supp1 = length(norms_u);
    
    %% Sparsify solution more
    if sparsification
        supp = size(uk.u, 2) + 1;
        while supp > size(uk.u, 2)
          supp = size(uk.u, 2);
          Kred = p.k(p, p.xhat, uk.x);
          uk = sparsify(uk, Kred'*Kred, N);
        end
    end

    %% optimize x
    if optimize_x
        %uk = p.optimize_x(p, y_ref, uk);
        uk = p.optimize_xu(p, y_ref, alpha, phi, uk);

        uk_x = uk;
    end
    
    %% Solve subproblem
    uk = p.optimize_u(p, y_ref, alpha, phi, uk);
    
    norms_u = computeNorm(uk.u, N);
    supp_ind = find(norms_u > 0);
    uk.u = uk.u(:,supp_ind);
    uk.x = uk.x(:,supp_ind);
    
    % save old j
    oldj = j;

    % updated values
    Ku = p.K(p, p.xhat, uk);
    norms_u = computeNorm(uk.u, N);
    j = obj.F(Ku - y_ref) + alpha*sum(phi.phi(norms_u));
    suppsize = nnz(norms_u);

    %% Output
    fprintf('PDAP: %3i, desc: (%1.0e,%1.0e), supp: %i->%i, j: (%1.2e,%1.0e), M0: %1.2e\n', ...
            iter, oldj - newj1, newj1 - j, supp1, suppsize, j, upperb, M0);

    % save diagnostics
    iter = iter+1;
    alg_out.us{iter} = uk;
    alg_out.js(iter) = j;
    alg_out.supps(iter) = suppsize;
    alg_out.tics(iter) = toc;
    alg_out.Psis(iter-1) = upperb;
    
    % plotting
    if mod(iter, plot_every) == 0
        yk = obj.dF(Ku - y_ref);
        figure(2001);
        p.plot_forward(p, uk, y_ref);
        figure(2002);
        p.plot_adjoint(p, uk, yk, alpha);

        %if optimize_x
        %    figure(2003);
        %    p.plot_adjoint(p, uk_x, obj.dF(p.K(p, p.xhat, uk_x) - y_ref), alpha);
        %    title('optimize positions');
        %end
        drawnow;
    end

end

%% return solution
u_opt = uk;

if plot_final
    yk = obj.dF(Ku - y_ref);
    figure(2001);
    p.plot_forward(p, uk, y_ref);
    figure(2002);
    p.plot_adjoint(p, uk, yk, alpha);
    drawnow;
end

end
