
p = setup_problem_NN_stereo_RePU(2);

is_octave = exist('OCTAVE_VERSION', 'builtin');
if is_octave
  pkg load optim;
end

%f_d = @(x) 2*exp(-50*(x).^2) + 0.2*sin(4*pi*x) + 4*max(x-0.5,0) + 0.3*(x<=-0.5);
%f_d = @(x) 2*exp(-50*(x).^2) + 0.2*sin(4*pi*x) + 4*max(x-0.5,0);
%f_d = @(x) abs(sin(2*x).*exp(-5*(x).^2/2)).^2;
%f_d = @(x) ones(size(x)) + x/.1;
%f_d = @(x) abs(sin(7*(0.001+abs(x).^2).^(1/4)).*exp(-2*(x).^2/2)).^2;
%f_d = @(x) abs(sin(7*(1+abs(x).^2).^(1/2)).*exp(-(x).^2/2));

f_d = @(x) cos(10*(.001+x.^2).^(1/8));

y_d = f_d(p.xhat)';

y_d = y_d + 0.05*randn(size(y_d));

alpha = .000001;

gamma = 0;
phi = p.Phi(p, gamma);

alg_opts = struct();
alg_opts.max_step = 15;
alg_opts.plot_every = 1;
alg_opts.optimize_x = true;
alg_opts.sparsification = true;
alg_opts.TOL = 1e-6;

[u_l1, alg_l1] = PDAPmultisemidiscrete(p, y_d, alpha, phi, alg_opts);

figure(1);
u_l1_pp = p.postprocess(p, u_l1, 1e-5);
p.plot_adjoint(p, u_l1_pp, p.obj.dF(p.K(p, p.xhat, u_l1_pp)-y_d), alpha)
figure(2);
p.plot_forward(p, u_l1_pp, y_d)
drawnow;

Nnodes_l1 = length(u_l1_pp.x)
l2_err_l1 = sqrt(2*p.obj.F(p.K(p, p.xhat, u_l1_pp)-y_d))


%% nonconvex problem
gamma = 1;
phi = p.Phi(p, gamma);
%alg_opts.u0 = u_l1;
[u_opt, alg_out] = PDAPmultisemidiscrete(p, y_d, alpha, phi, alg_opts);

figure(3);
p.plot_adjoint(p, u_opt, p.obj.dF(p.K(p, p.xhat, u_opt)-y_d), alpha)
figure(4);
p.plot_forward(p, u_opt, y_d)

Nnodes_phi = length(u_opt.x)
l2_err_phi = sqrt(2*p.obj.F(p.K(p, p.xhat, u_opt)-y_d))
