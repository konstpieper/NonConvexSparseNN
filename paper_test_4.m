% force coefficients to be on upper hemisphere?
force_upper = false;

dim = 3;

p = setup_problem_NN_xd(dim, .01, force_upper);

is_octave = exist('OCTAVE_VERSION', 'builtin');
if is_octave
  pkg load optim;
end

f_d = @(x) sum(x(2:end,:).^2./(x(1,:)+1.05), 1) + 2*(x(1,:)+1.05).^4;
y_d = f_d(p.xhat)';

alg_opts = struct();
alg_opts.max_step = 500*100;
alg_opts.plot_every = 0;
alg_opts.print_every = 50;
alg_opts.plot_final = 0;
alg_opts.sparsification = false;
alg_opts.TOL = 1e-4;

alpha = .01;
alphas = alpha*[1, 1/4, 1/16, 1/64];

gamma = 0;
phi = p.Phi(p, gamma);

u_l1 = cell(length(alphas),1);
alg_out_1 = cell(length(alphas),1);
Nnodes_l1 = cell(length(alphas),1);
l2_err_l1 = cell(length(alphas),1);

uinit = p.u_zero;

for n = 1:length(alphas)
  alpha = alphas(n);

  alg_opts.u0 = uinit;
  %[u_l1{n}, alg_out_1{n}] = PDAPmultisemidiscrete(p, y_d, alpha, phi, alg_opts);
  [u_l1{n}, alg_out_1{n}] = solve_TV_GNAP(p, y_d, alpha, phi, alg_opts);

  uinit = u_l1{n};

  figure(1);
  p.plot_adjoint(p, u_l1{n}, p.obj.dF(p.K(p, p.xhat, u_l1{n})-y_d), alpha)
  %matlab2tikz('paper_test_4/l1_adjoint.tikz')
  figure(2);
  p.plot_forward(p, u_l1{n}, y_d)
  %matlab2tikz('paper_test_4/l1_forward.tikz')
  drawnow;

  Nnodes_l1{n} = length(u_l1{n}.x)
  l2_err_l1{n} = sqrt(2*p.obj.F(p.K(p, p.xhat, u_l1{n})-y_d))
end

gamma = 5;
phi = p.Phi(p, gamma);

alg_opts.optimize_x = true;
alg_opts.sparsification = true;

u_opt = cell(length(alphas),1);
alg_out = cell(length(alphas),1);
Nnodes_phi = cell(length(alphas),1);
l2_err_phi = cell(length(alphas),1);

uinit = p.u_zero;

for n = 1:length(alphas)
  alpha = alphas(n);

  alg_opts.u0 = uinit;
  %[u_opt{n}, alg_out{n}] = PDAPmultisemidiscrete(p, y_d, alpha, phi, alg_opts);
  [u_opt{n}, alg_out{n}] = solve_TV_GNAP(p, y_d, alpha, phi, alg_opts);

  uinit = u_opt{n};

  figure(3);
  p.plot_adjoint(p, u_opt{n}, p.obj.dF(p.K(p, p.xhat, u_opt{n})-y_d), alpha)
  %matlab2tikz('paper_test_4/phi_adjoint.tikz')
  figure(4);
  p.plot_forward(p, u_opt{n}, y_d)
  %matlab2tikz('paper_test_4/phi_forward.tikz')
  drawnow;

  Nnodes_phi{n} = length(u_opt{n}.x)
  l2_err_phi{n} = sqrt(2*p.obj.F(p.K(p, p.xhat, u_opt{n})-y_d))
end

