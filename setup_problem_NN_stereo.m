function p = setup_problem_NN_stereo(delta, force_upper)
#{

input 
  delta - smoothing parameter
  
output 
  p - a structure with functions and data of the model, contains
      .delta - same as above
      .N - the argument dimension
      .Omega - the interval wehere the function is sampled
      .u_zero - zero measure used for initialization
      .xhat - sampling locations (of the target function) 
      .K - returns the value of network and its derivatives at p.xhat
      .k - value of the ReLu and its derivative at p.xhat
      .plot_forward - plots the true and  estimated values at current iteration
      .Phi - the penulty function, its first and second derivatives
      .obj - contains functions which return the l2-mean, mean of an array
      .Ks -  same as K but the input is entered diffrently 
      .find_max - find max of dual variable at multiple points 
      .optimize_u - fixed nodes, optimizes the outer weights
      .optimize_xu - optimize both over nodes and outer weights
      .plot_adjoint - plots the adjoint variable at given iteration
      .postprocess - sorts the x, replaces clustered entries that are closer than pp_radius with weighted average of same energy      
#}

p = struct();

%% smoothing parameter for the actiation function \sigma
p.delta = delta;

%% scalar problem (one dimensional output)
p.N = 1;

% input dimension
p.dim = 1;

%% size of the domain D = [-R,R]
p.R = 1.;

%% size of the ball in the projecive plane that leads to nonzero activation functions
RO = p.R + sqrt(1+p.R^2);
p.Omega = [-RO, RO];

%% force coefficients to be on the upper hemisphere
%% and use that absolute value activation function \sigma(y) = |y|
%% instead of \sigma(y) = \max(y,0)
p.force_upper = force_upper;

%% zero measure
p.u_zero = struct('x', zeros(1,0), 'u', zeros(p.N,0));

%% observation points in D
Npoints = 1000;
%p.xhat = [0,sort(-p.R + 2*p.R*rand(1,Npoints -2)),1];
p.xhat = linspace(-p.R, p.R, Npoints);
%p.xhat = sort((p.R/3)*randn(1, Npoints));

%% kernel functions
p.K = @K;
p.k = @kernel;
p.plot_forward = @plot_forward;

%% stuff for optimization
p.Phi = @Phi;
p.obj = Tracking(p);

p.Ks = @Ks;
p.find_max = @find_max;
p.optimize_u = @optimize_u;
p.optimize_xu = @optimize_xu;

p.plot_adjoint = @plot_adjoint;
p.postprocess = @postprocess;

end


function [Ku, dKu] = K(p, xhat, u)

  %% evaluate the K*u at the points xhat
  %% K <-> N
  %% u <-> \mu
  %% u.x <-> \omega
  %% u.u <-> c

  [k, dk] = kernel(p, xhat, u.x);

  Ku = k * u.u(:);
  dKu = dk * u.u(:);
  %ddKu = ddk * u.u(:);

end

function [Ksy, dKsy] = Ks(p, x, xhat, y)

  %% evaluate the \sum_{\xi\in xhat} K^\star(\xi) y(\xi) at the points x
  
  [k, dk] = kernel(p, xhat, x);

  Ksy = k' * y;
  dKsy = dk' * y;
  %ddKsy = ddk' * y;

end

function [k, dk] = kernel(p, xhat, x)
%{
Inout
  p - the model in current state
  xhat - target function sampling location
  x - the sterographic coordinates of the nodes
Output
  k - the value of the ReLU at xhat
  dk -  differential of k at xhat
%}

%% Evaluate the kernel k(xhat, x) = \sigma(a(x)*xhat + b(x), 0),
%% where \omega(x) = (a(x), b(x)) \in Sd is parametrized in terms of
%% the stereographic projection x = stereo(a,b). 
  
  % take care of zero measure
  if isempty(x)
    X = zeros(numel(xhat),0);
    Xhat = zeros(numel(xhat),0);
  else
    [X, Xhat] = meshgrid(x, xhat);
  end

  %% x = stereo(a,b) is the stereograpic projection from the south pole S=(a,b)=(0,-1)
  opx2 = 1 + X.^2;

  b = (2 - opx2) ./ opx2;
  a = 2 * X ./ opx2;

  dbdx = - 2 * a ./ opx2;
  %- 4 * x ./ (1 + x2).^2;
  dadx = 2 * b ./ opx2;
  % 2 * (1 - x2) ./ (1 + x2).^2;

  %% y = a*xhat + b
  y = a.*Xhat + b;
  dydx = dadx.*Xhat + dbdx;
  %ddydxx = ddadxx.*Xhat + ddbdxx;

  % smoothing parameter for max
  delta = p.delta;

  %% kernel = max_delta(0, y)
  absy = sqrt(delta^2 + y.^2);
  %absa = sqrt(0.01^2 + a.^2);

  if ~p.force_upper
    k = (1/2) * (absy + y);
    dk = (1/2) * (y ./ absy + 1) .* dydx;

    %k = (1/2) * (absy + y) ./ absa;
    %dk = (1/2) * ((y ./ absy + 1) ./ absa .* dydx - (absy + y) .* (a./absa) .* dadx ./ absa.^2);
    %ddk = (1/2) * (1 - (y ./ smooth_abs).^2) ./ smooth_abs .* ddydxx;
  else
    k = (1/2) * absy;
    dk = (1/2) * (y ./ absy) .* dydx;

    %k = (1/2) * absy ./ absa;
    %dk = (1/2) * ((y ./ absy) .* dydx ./ absa - absy .* (a./absa) .* dadx ./ absa.^2);
  end
end

function phi = Phi(p, gamma)

  %% set up the cost function \phi, derivatives, inverse and the prox operator

  phi = struct();
  phi.gamma = gamma;

  if gamma == 0
    phi.phi = @(t) t;
    phi.dphi = @(t) ones(size(t));
    phi.ddphi = @(t) zeros(size(t));
    phi.inv = @(y) y;

    phi.prox = @(sigma, g) max(g - sigma, 0);
  else
    th = 1/2;
    gam = gamma/(1-th);

    phi.phi = @(t) th * t + (1-th) * log(1 + gam * t) / gam;
    phi.dphi = @(t) th + (1-th) ./ (1 + gam * t);
    phi.ddphi = @(t) - (1-th) * gam ./ (1 + gam * t).^2;
                         %phi.inv = @(y) (exp(gamma * y) - 1) / gamma;
    phi.inv = @(y) y / th;

    phi.prox = @(sigma, g) (1/2)*(g - sigma*th - 1/gam + sqrt( (g - sigma*th - 1/gam).^2 + 4*(g - sigma)/gam)) .* (g>=sigma);
  end

%phi.val = @(t) abs(t) - min(max(1/(gamma), abs(t)) - 1/(2*gamma), (gamma/2) * t.^2);
%phi.dma = @(t) - max(-1, min(1, gamma * t));
%phi.ddma = @(t) - gamma * (abs(t) <= 1/gamma);

end

function obj = Tracking(p)

  %% set up the loss or tracking function F(y) and derivatives

  Nx = numel(p.xhat);
  
  obj.F = @(y) 1/(2*Nx) * norm(y)^2;
  obj.dF = @(y) 1/Nx * y;
  obj.ddF = @(y) 1/Nx;

  %p = 5;
  %obj.F = @(y) 1/(p) * norm(y, p)^p;
  %obj.dF = @(y) 1 * sign(y).*abs(y).^(p-1);
  %obj.ddF = @(y) 1 * spdiags((p-1)*abs(y).^(p-2), 0, Nx, Nx);

end

function u_pp = postprocess(p, u, pp_radius)

  %% take a given discrete measure u and lump together Dirac delta functions
  %% that are less than pp_radius apart
  
  [x, perm] = sort(u.x);
  u = u.u(perm);
  
  cut = [0, find(diff(x) > pp_radius), length(x)];
  U = zeros(1, length(cut)-1);
  X = zeros(1, length(cut)-1);
  for cp = 1:length(cut)-1
    range = cut(cp)+1:cut(cp+1);
    U(1,cp) = sum(u(1,range));
    X(1,cp) = sum(x(1,range).*abs(u(1,range))) / sum(abs(u(1,range)));
  end
  
  u_pp = struct('x', X, 'u', U);    
end

function plot_forward(p, u, y_d)

  %% plot the measure and the function and its approximation
  pO = p.Omega;
  if p.force_upper
    pO = [-1,1];
  end
  
  %unstereo = @(x) atan2(2*x./(1+x.^2), (1-x.^2)./(1+x.^2));
  %pO = [-pi, pi];
  
  unstereo = @(x) min(10,max(-10, -(1-x.^2)./(2*x) ));
  pO = [-1.2, 1.2];
  
  %% plot measure

  subplot(2,1,1);

  xi = [u.x; u.x];
  yi = [zeros(size(u.u)); u.u];
  plot(unstereo(xi), yi, 'k', 'LineWidth', 1);
  hold on;
  plot(pO, [0,0], 'k--', 'LineWidth', 1);
  plot(unstereo(u.x), u.u, 'ko', 'LineWidth', 3, 'MarkerSize', 3);
  set(gca, 'FontSize', 12);
  xlabel('x_n = -b_n/a_n')
  ylabel('c_n')
  hold off;
  axis([pO, min(-.01,min([u.u,0]))*1.2, max(.01,max([u.u,0]))*1.2]);

  %% plot function

  subplot(2,1,2);

  t = linspace(1.2*min(p.xhat), 1.2*max(p.xhat), 1000);
  plot(p.xhat, y_d, '--', 'LineWidth', 1, 'Color', [0 0.4470 0.7410]);
  hold on;
  Ku = reshape(p.K(p, t, u), [p.N, length(t)]);
  plot(t, Ku, 'k', 'LineWidth', 2);
  Kxu = reshape(p.K(p, unstereo(u.x), u), [p.N, length(u.x)]);
  plot(unstereo(u.x), Kxu, 'rx', 'LineWidth', 2, 'MarkerSize', 10, 'Color', [0.8500 0.3250 0.0980]);
  axis([min(t), max(t), min(-.1,min(y_d))*1.2, max(.1,max(y_d))*1.2])
  set(gca, 'FontSize', 12);
  xlabel('x')
  ylabel('y')
  hold off;

end

function plot_adjoint(p, u, y, alpha, pp_radius)

  %% plot the measure and the dual variable corresponding to y
  
  if nargin == 5
    u = postprocess(p, u, pp_radius);
  end

  pO = p.Omega;
  if p.force_upper
    pO = [-1,1];
  end
  t = [-1000, linspace(pO(1), pO(2), 1000), 1000];

  unstereo = @(x) atan2(2*x./(1+x.^2), (1-x.^2)./(1+x.^2));
  pO = [-pi,pi];
  
  %% plot measure

  subplot(2,1,1);

  xi = [u.x; u.x];
  yi = [zeros(size(u.u)); u.u];
  plot(unstereo(xi), yi, 'k', 'LineWidth', 1);
  hold on;
  plot(pO, [0,0], 'k--', 'LineWidth', 1);
  plot(unstereo(u.x), u.u, 'ko', 'LineWidth', 3, 'MarkerSize', 3);
  set(gca, 'FontSize', 12);
  hold off;
  axis([pO, min(-.01,min(u.u))*1.2, max(.01,max(u.u))*1.2]);

  %% plot dual variable

  subplot(2,1,2);

  Ksy = reshape(p.Ks(p, t, p.xhat, y), [p.N, length(t)]);
  plot(unstereo(t), Ksy, 'k', 'LineWidth', 2);
  hold on;
  Ksysupp = reshape(p.Ks(p, u.x, p.xhat, y), [p.N, length(u.x)]);
  plot(unstereo(u.x), Ksysupp, 'ko', 'LineWidth', 1, 'MarkerSize', 5);
  plot(pO, [alpha, alpha], 'k--', 'LineWidth', 1);
  plot(pO, [-alpha, -alpha], 'k--', 'LineWidth', 1);
  axis([pO, min(-alpha,min(Ksy))*1.2, max(alpha,max(Ksy))*1.2])
  set(gca, 'FontSize', 12);
  hold off;

end

function u_opt = optimize_u(p, y_d, alpha, phi, u)

  %% compute the optimal coefficients of the measure for fixed locations
  %% solve the finite dim. problem
  %%   min_{c \in \R^N} F(KK c - y_d) + \Phi(c)  
  %% where KK is the matrix KK_{i,j} = k(xhat_i, x_j)
  
  Kred = p.k(p, p.xhat, u.x);
  ured = u.u(:);
  
  %% Solve finite dimensional subproblem with semismooth Newton
  %% tolerance is set to machine precision
  ured = SSN(p, Kred, y_d, alpha, phi, ured, p.N);
  %ured = ProxGradPhi(p, Kred, y_d, alpha, phi, ured, p.N);
  %ured = SSN_TR(p, Kred, y_d, alpha, phi, ured, p.N);

  u_opt = u;
  u_opt.u = ured';

end

function xmax = find_max(p, y, x0)

  %% find the maxima of the absolute value of the dual variable
  %% max_{\omega} |p(\pmega)| = |<F'(y), \sigma(\omega)>|

  %% xmax will contain a number of local maxima (including hopefully the global one)

TOL = 1e-6;

% initial guesses
Nguess = 50;

if size(x0,2) > Nguess/2
  ii = randperm(size(x0,2));
  x0 = x0(:,ii(1:floor(Nguess/2)));
end
Nguess = Nguess - size(x0,2);

%% throw some random points into parameter space
randomb = sort(min(p.xhat) + (max(p.xhat) - min(p.xhat))*rand(1, Nguess));
randomx = +1 ./ sqrt(1 + randomb.^2);
randomb = randomb ./ sqrt(1 + randomb.^2);
randomx = randomx ./ (1 + randomb);
randomx = sign(rand(size(randomx))-1/2) .* randomx;
%randomx = pO(1) + (pO(2) - pO(1))*rand(1,Nguess);
x0 = [0, randomx, x0];

y_norm = y/norm(y);
j = @(x) optfun_vJ(p, y_norm, x);

ng = 100;
iter = 0;
while ng > 1e4*TOL && iter < 10
  
  %% call the local optimizer
  opts = optimset('Algorithm', 'quasi-newton', 'GradObj', 'on', 'Display', 'off', ...
                  'TolFun', 1e-30, 'TolX', TOL, 'MaxIter', 500, 'MaxFunEvals', 10000);
  [xmax, ~, cvg] = fminunc(j, x0(:), opts);

  [jmax, g] = j(xmax);
  ng = norm(g);
  %check_gradient(p, xmax', j);

  %% post-process the solution
  xmax = xmax';
  %% eliminate small local minima
  norm_grad = abs(Ks(p, xmax, p.xhat, y));
  xmax = xmax(norm_grad > (1/4)*max(norm_grad));
  %% eliminate duplicates
  u_pp = p.postprocess(p, struct('x', xmax, 'u', ones(1,length(xmax))), 1e2*TOL);

  xmax = u_pp.x;
  x0 = xmax;
  
  fprintf('\tpos: val: %f -> %f, |g|=%f (%i)\n', j(x0(:)), j(xmax), norm(g), cvg);

  iter = iter+1;
end

end

function [j, g] = optfun_vJ(p, y, x)
  %% optimization function for find_max including gradient
  
  [Ksy, dKsy] = Ks(p, x, p.xhat, y);
  Ksy = reshape(Ksy, [p.N, length(x)]);
  dKsy = reshape(dKsy, [p.N, length(x)]);
  normK2 = sum(abs(Ksy).^2, 1);
  j = - sum(normK2, 2) / 2;
  g = - sum(real(Ksy.*conj(dKsy)), 1)';
end

function check_gradient(p, x, j, dj)
  %% function to verify the correctness of the gradient implementation

  if nargin > 3
    gx = dj(x);
  else
    [~, gx] = j(x);
  end
  
  dx = rand(size(x));
  tau = 2.^[0:-1:-30];
  err = zeros(size(tau));
  for ntau = 1:length(tau)
    jp = j(x + tau(ntau)*dx);
    jm = j(x - tau(ntau)*dx);
    fin_diff = (jp - jm)/(2*tau(ntau));
    err(ntau) = abs(gx'*dx - fin_diff);
  end
  figure(666);
  loglog(tau, err);
  drawnow;
  
end

function u_opt = optimize_xu(p, y_d, alpha, phi, u)

  %% compte a local minimum of the training problem
  %% \min_{c, \omega \in Sd} F(N_{c,\omega}) + \sum \phi(c)
  %% (not needed for convergence of the algorithm, but improves quality of solutions)
  %%  naming convention \omega = (a(x), b(x)); c = u;  

  [dim, Nd] = size(u.x);

  % initial guesses
  xu0 = [u.x(:); u.u(:)];

  % upper and lower bounds and (fixed) signum
  ugeq0 = (u.u >= 0);
  ul = -Inf(Nd,1); ul(ugeq0) = 0;
  uu = Inf(Nd,1); uu(~ugeq0) = 0;
  xul = [-Inf(Nd,1); ul];
  xuu = [Inf(Nd,1); uu];

  signu = -ones(Nd,1);
  signu(ugeq0) = 1;

  is_octave = exist('OCTAVE_VERSION', 'builtin');

  if ~is_octave
    %% MATLAB
    j  = @(xu) optfun_vJxu(p, y_d, alpha, phi, signu, xu);
    
    opts = optimset('Algorithm', 'trust-region-reflective', ...
                    'GradObj', 'on', 'Hessian', 'off', ...
                    'Display', 'off', 'DerivativeCheck', 'off', 'FinDiffType', 'central', ...
                    'TolFun', 1e-30, 'TolX', 1e-5, 'MaxIter', 100, 'MaxFunEvals', 1000);

    xumin = fmincon(j, xu0, [], [], [], [], xul, xuu, [], opts);
    [jmin, g] = j(xumin);
    inactive = (xumin < xuu) & (xumin > xul);
    fprintf('\toptimized points and coeff: val: %f -> %f, |g|=%f\n', j(xu0), jmin, norm(g(inactive)));

  else
    %% OCTAVE
    j  = @(xu) optfun_vxu(p, y_d, alpha, phi, signu, xu);
    dj = @(xu) optfun_Jxu(p, y_d, alpha, phi, signu, xu);

    opts = optimset('Algorithm', 'lm_feasible', 'lb', xul, 'ub', xuu, ...
                    'TolFun', 1e-30, 'objf_grad', dj, 'MaxIter', 500 );
    
    [xumin, minval, cvg] = nonlin_min(j, xu0, opts);

    inactive = (xumin < xuu) & (xumin > xul);
    g = dj(xumin);
    ng = norm(g(inactive));
    if cvg > 0
      fprintf('\toptimized points: val: %f -> %f, |g|=%f, status=%i\n', j(xu0), minval, ng, cvg)
    else
      fprintf('\tfailed opt. points: val: %f -> %f, |g|=%f, status=%i\n', j(xu0), minval, ng, cvg)
    end
  end
  
  %check_gradient(p, xmax, j, dj);

  u_opt = u;
  u_opt.x = xumin(1:end/2)';
  u_opt.u = xumin(end/2+1:end)';

  if p.force_upper
    norm_x2 = sum(u_opt.x.^2, 1);
    u_opt.x(:,norm_x2>1) = -u_opt.x(:,norm_x2>1)./norm_x2(:,norm_x2>1);
  end

end

function [j,g] = optfun_vJxu(p, y_d, alpha, phi, signu, xu)
  %% optimization function for optimize_xu including gradient

  x = xu(1:end/2);
  u = xu(end/2+1:end);

  [K, dK] = kernel(p, p.xhat, x);
  y = K*u;
  j = p.obj.F(y - y_d) + alpha*sum(phi.phi(signu.*u));

  if nargout > 1
    gF = p.obj.dF(y - y_d);
    g = [u.*(dK'*gF); K'*gF + alpha*(signu.*phi.dphi(signu.*u))];
  end

end

function j = optfun_vxu(p, y_d, alpha, phi, signu, xu)

  x = xu(1:end/2);
  u = xu(end/2+1:end);

  [K, dK] = kernel(p, p.xhat, x);
  y = K*u;
  j = p.obj.F(y - y_d) + alpha*sum(phi.phi(signu.*u));

end

function g = optfun_Jxu(p, y_d, alpha, phi, signu, xu)

  x = xu(1:end/2);
  u = xu(end/2+1:end);

  [K, dK] = kernel(p, p.xhat, x);
  y = K*u;

  gF = p.obj.dF(y - y_d);
  g = [u.*(dK'*gF); K'*gF + alpha*(signu.*phi.dphi(signu.*u))];

end
