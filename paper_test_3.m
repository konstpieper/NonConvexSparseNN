% force coefficients to be on upper hemisphere?
force_upper = false;

p = setup_problem_NN_2d(.001, force_upper);

is_octave = exist('OCTAVE_VERSION', 'builtin');
if is_octave
  pkg load optim;
end

f_d = @(x) sqrt(sum((x - 0.1).^2,1));
y_d = f_d(p.xhat)';

alg_opts = struct();
alg_opts.max_step = 15;
alg_opts.plot_every = 5;
alg_opts.sparsification = false;
alg_opts.TOL = 1e-6;
alg_opts.optimize_x = false;

alpha = .00001;

gamma = 0;
phi = p.Phi(p, gamma);

[u_l1, alg_out_1] = PDAPmultisemidiscrete(p, y_d, alpha, phi, alg_opts);


figure(1);
u_l1_pp = p.postprocess(p, u_l1, 1e-3);
p.plot_adjoint(p, u_l1_pp, p.obj.dF(p.K(p, p.xhat, u_l1_pp)-y_d), alpha)
%matlab2tikz('paper_test_3/l1_adjoint.tikz')
figure(2);
p.plot_forward(p, u_l1_pp, y_d);
%matlab2tikz('paper_test_3/l1_forward.tikz')
drawnow;

Nnodes_l1 = length(u_l1_pp.x)
l2_err_l1 = sqrt(2*p.obj.F(p.K(p, p.xhat, u_l1_pp)-y_d))

%% nonconvex problem
gammas = {1e-3,1e-2,1e-1,1,10};

u_opt = cell(length(gammas),1);
alg_out = cell(length(gammas),1);
Nnodes_phi = cell(length(gammas),1);
l2_err_phi = cell(length(gammas),1);

for n = 1:length(gammas)
  gamma = gammas{n};
  phi = p.Phi(p, gamma);
                                %alg_opts.u0 = u_l1;
  [u_opt{n}, alg_out{n}] = PDAPmultisemidiscrete(p, y_d, alpha, phi, alg_opts);

  figure(3);
  p.plot_adjoint(p, u_opt{n}, p.obj.dF(p.K(p, p.xhat, u_opt{n})-y_d), alpha)
  %matlab2tikz('paper_test_3/phi_adjoint.tikz')
  figure(4);
  p.plot_forward(p, u_opt{n}, y_d)
  %matlab2tikz('paper_test_3/phi_forward.tikz')
  drawnow;
  
  Nnodes_phi{n} = length(u_opt{n}.x)
  l2_err_phi{n} = sqrt(2*p.obj.F(p.K(p, p.xhat, u_opt{n})-y_d))
end
