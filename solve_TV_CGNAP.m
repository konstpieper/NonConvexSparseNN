function [u_opt, alg_out] = solve_TV_CGNAP(p, y_ref, alpha, phi, alg_opts)

N = p.N;
assert(N==1);
dim = p.dim;  
obj = p.obj;

Ndata = length(y_ref);

%% setup options field
if (nargin <= 3)
    alg_opts = struct();
end

max_step = get_field_default(alg_opts, 'max_step', 1000);
TOL = get_field_default(alg_opts, 'TOL', 1e-5);

plot_final = get_field_default(alg_opts, 'plot_final', true);
plot_every = get_field_default(alg_opts, 'plot_every', 0);
print_every = get_field_default(alg_opts, 'print_every', 20);

blocksize = get_field_default(alg_opts, 'blocksize', 50);

% initial guess
u0 = get_field_default(alg_opts, 'u0', p.u_zero);
uk = u0;

% initial values
[K_big, dK_big] = p.k(p, p.xhat, uk.x);

yk = K_big * uk.u(:);
norms_c = computeNorm(uk.u, N);
j = obj.F(yk - y_ref) / alpha + sum(phi.phi(norms_c));
suppsize = nnz(norms_c);

% save algorithmic diagnostics
alg_out = struct();
alg_out.us{1} = uk;
alg_out.js = j;
alg_out.supps = suppsize;
alg_out.tics = 0;
alg_out.Psis = 0;


tic;

ck = uk.u;
xk = uk.x;

shape_x = @(v) reshape(v, dim, []);
shape_dK = @(dK) reshape(permute(dK, [1,3,2]), Ndata, []);

%% active set
Nc = length(ck);
blcki = randperm(Nc, min(blocksize, Nc));
Ncblck = length(blcki);

fprintf('GNAP iter: 0, j=%f, supp=%d\n', j, Nc);

%% change to Robinson variables
qk = sign(ck) + ck;
Prox = @(v) computeProx(v, 1, N);
assert(norm(ck - Prox(qk), inf) < 1e-14);
ck = Prox(qk);

Dphima = @(c) (phi.dphi(abs(c)) - 1).*sign(c);
DDphima = @(c) phi.ddphi(abs(c));

yk = K_big * ck(:);
misfit = yk - y_ref;
norms_c = computeNorm(ck, N);
j = obj.F(misfit) / alpha + sum(phi.phi(norms_c));

theta_old = 1;

for k = 1:max_step
  
  Gp = [K_big(:,blcki), shape_dK(dK_big(:,blcki,:) .* ck(blcki))];
  R = (1/alpha) * (Gp' * obj.dF(misfit)) + [Dphima(ck(blcki))' + (qk(blcki) - ck(blcki))'; zeros(Ncblck*dim,1)];

  SI = obj.ddF(misfit);
  II = Gp' * SI * Gp;
  
  %kp = shape_dK(dK)' * obj.dF(misfit)
  kpp = 3 * norm(obj.dF(misfit), 1) * reshape(sqrt(eps) + ones(dim,1)*abs(ck(blcki)), [], 1);
  Icor = [ zeros(Ncblck,Ncblck),     zeros(Ncblck,dim*Ncblck);
           zeros(dim*Ncblck,Ncblck),                diag(kpp) ];
  
  %II = (Gp' * SI * Gp);
  %Icor = beta * diag([zeros(Nc,1); ones(Nc,1)]);
  
  HH = (1/alpha) * (II + Icor);

  %% SSN correction  
  %DP = diag([abs(qk) > 1; ones(Nc,1)]);
  DP = diag([abs(qk(blcki)') >= 1; reshape(ones(dim,1)*(abs(ck(blcki)) > 0), [], 1)]);
  DDphi = 0*diag([DDphima(ck(blcki)), zeros(1,dim*Ncblck)]);
  DR = HH * DP + DDphi + (eye((1+dim)*Ncblck) - DP);
  
  dz = - DR \ R;

  qold = qk;
  xold = xk;
  jold = j;
  theta = min(theta_old * 2, 1 - 1e-14);
  has_descent = false;
  while ~has_descent && theta > 1e-16
    qk(:,blcki) = qold(:,blcki) + theta * reshape(dz(1:Ncblck), 1, []);
    xk(:,blcki) = xold(:,blcki) + theta * reshape(dz(Ncblck+1:end), dim, []);
    ck = Prox(qk);
    [K, dK] = p.k(p, p.xhat, xk(:,blcki));

    K_big(:,blcki) = K;
    dK_big(:,blcki,:) = dK;

    yk = K_big * ck(:);
    misfit = yk - y_ref;
    norms_c = computeNorm(ck, N);
    j = obj.F(misfit) / alpha + sum(phi.phi(norms_c));
    descent = j - jold;
    pred = theta * (R' * (DP * dz));
    %pred / descent
    has_descent = (descent <= (pred + 1e-11) / 3);
    if ~has_descent
      theta = theta / 1.5;
    end
  end
  theta_old = theta;

  %% active set
  suppc = (abs(qk) > 1);

  %% constraint violation
  %% generate search grid
  [a, b] = sample_weights_TN(p.xhat, p.R, 10+blocksize);
  a = a./sqrt(1 + b.^2);
  b = b./sqrt(1 + b.^2);
  omegas_new = a ./ (1 + b);
  if k > 1
    omegas = [omegas(:,ind_max_sh_eta), omegas_new];
  else
    omegas = omegas_new;
  end
  K_test = p.k(p, p.xhat, omegas);

  
  eta = 1/alpha * K_test' * obj.dF(misfit);
  sh_eta = abs(Prox(eta));
  [max_sh_eta, ind_max_sh_eta] = max(sh_eta);
  
  if mod(k, print_every) == 0
    fprintf('CGNAP iter: %i, j=%f, supp=(%d->%d), desc=%1.1e, dz: %1.1e, viol=%1.1e, theta: %1.1e\n', ...
            k, j, Nc, sum(suppc), descent, norm(dz, Inf), max_sh_eta, theta);
  end

  if ~has_descent
    %% linesearch failed, should not happen
    keyboard
  end

  %% prune zero coefficient Diracs
  if any(~suppc)
    Nc = sum(suppc);
    qk(:,~suppc) = [];
    xk(:,~suppc) = [];
    K_big(:,~suppc) = [];
    dK_big(:,~suppc,:) = [];
    ck = Prox(qk);
  end

  %% try adding promising new zero coeffs
  grad_supp_c = 1/alpha * (K_big' * obj.dF(misfit)) + Dphima(ck)' + (qk - ck)';
  tresh_c = abs(grad_supp_c)';
  grad_supp_y = 1/alpha * shape_dK(dK_big.*ck)' * obj.dF(misfit);
  tresh_y = sqrt(sum(reshape(grad_supp_y, dim, []).^2));

  tresh = (tresh_c + 0.01*tresh_y);
  
  %% new block indices
  [tresh, ind_th] = sort(tresh, 'descend');
  blcki = ind_th(1:min(blocksize, Nc));
  Ncblck = length(blcki);
  
  if max_sh_eta > 2*norm(tresh, Inf)
    Nc = Nc+1;
    qk = [qk, -sign(eta(ind_max_sh_eta))];
    ck = [ck, 0];
    xk = [xk, omegas(:,ind_max_sh_eta)];
    [K, dK] = p.k(p, p.xhat, xk(:, Nc));
    K_big(:, Nc) = K;
    dK_big(:, Nc, :) = dK;
    
    fprintf('  insert: viol=%1.2e, |g_c|+|g_y|=%1.1e+%1.1e, supp:(%d->%d)\n', ...
            max_sh_eta, max(tresh_c), max(tresh_y), sum(suppc), Nc);
    
    if Ncblck < blocksize
      blcki = [blcki, Nc];
      Ncblck = length(blcki);
    else
      blcki(Ncblck) = Nc;
    end
  end

  blcki = sort(blcki);

  % save diagnostics
  uk = struct('u', ck, 'x', xk);
  alg_out.us{k} = uk;
  alg_out.js(k) = j;
  alg_out.supps(k) = Nc;
  alg_out.tics(k) = toc;
  
  if mod(k, plot_every) == 0
    figure(2001);
    p.plot_forward(p, uk, y_ref);
    figure(2002);
    p.plot_adjoint(p, uk, obj.dF(yk - y_ref), alpha);
    drawnow;
  end

  if abs(pred) / theta < TOL && norm(dz, Inf)/100 + max_sh_eta < TOL
    %% tolerance reached
    fprintf('CGNAP iter: %i, j=%f, supp=(%d->%d), desc=%1.1e, dz: %1.1e, viol=%1.1e, theta: %1.1e\n', ...
            k, j, Nc, sum(suppc), descent, norm(dz, Inf), max_sh_eta, theta);
    break
  end
end

%% undo Robinson variables
u_opt = uk;

if plot_final
  figure(2001);
  p.plot_forward(p, u_opt, y_ref);
  figure(2002);
  p.plot_adjoint(p, u_opt, obj.dF(yk - y_ref), alpha);
  drawnow;
end


end
