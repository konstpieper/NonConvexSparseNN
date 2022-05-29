% force coefficients to be on upper hemisphere?
force_upper = false;

dim = 3;

p = setup_problem_NN_xd(dim, .01, force_upper);

is_octave = exist('OCTAVE_VERSION', 'builtin');
if is_octave
  pkg load optim;
end

%f_d = @(x) sqrt(sum((x+.1).^2,1));
%f_d = @(x) cos(10*x(1,:).*x(2,:)) .* exp(-sum(4*x.^2,1)/2);
%f_d = @(x) max(0, 1 - 2*max(abs(x(1,:)), abs(x(2,:))));
%f_d = @(x) exp(- sum((3*x).^2, 1) / 2);

%kappa = 1000;
%softmax = @(a,b) (1/kappa) * (kappa*max(a,b) + log(exp(kappa*(a-max(a,b))) + exp(kappa*(b-max(a,b)))));
%f_d = @(x) softmax(0,softmax(x(1,:),x(2,:)));

%huber = @(t) (t <= 1/2).*(1/2).*t.^2 + (t > 1/2).*(1/2).*(t-1/4);
%f_d = @(x) huber(sqrt(sum((x+.1).^2, 1)));

f_d = @(x) sum(x(2:end,:).^2./(x(1,:)+1.05) + (x(1,:)+1.05).^4, 1);

%f_d = @(x) (x(1,:) >= 0) | (x(2,:) >= 0);

%f_d = @(x) sqrt(sum(x.^2, 1));
%f_d = @(x) sqrt(x(1,:).^2+x(2,:).^2);
y_d = f_d(p.xhat)';

%figure(10);
%Nomega = 5;
%t = linspace(0, 2*pi, 6*Nomega+1);
%a = [cos(t); sin(t)];
%R = (p.Omega(2) - p.Omega(1))/2;
%if ~p.force_upper
%  b = linspace(-sqrt(R), sqrt(R), 2*Nomega-1)';
%else
%  b = linspace(0, sqrt(R), Nomega)';
%end
%a1 = a(1,:)./sqrt(1 + b.^2);
%a2 = a(2,:)./sqrt(1 + b.^2);
%b = ones(size(a(1,:))) .* (b./sqrt(1 + b.^2));
%omegas = [a1(:)'; a2(:)'] ./ (1 + b(:)');
%Nomega = size(omegas,2);
%ul2 = struct('x', omegas, 'u', zeros(1,Nomega));
%Kred = p.k(p, p.xhat, ul2.x);
%ul2.u = ((1e-3*eye(Nomega) + Kred'*Kred)\(Kred'*y_d))';
%p.plot_forward(p, ul2, y_d);
%drawnow;

alg_opts = struct();
alg_opts.max_step = 10;
alg_opts.plot_every = 0;
alg_opts.sparsification = false;
alg_opts.TOL = 1e-6;

alpha = .01;

%gamma = 0;
%phi = p.Phi(p, gamma);

%[u_opt_1, alg_out_1] = PDAPmultisemidiscrete(p, y_d, alpha, phi, alg_opts);

gamma = 0;%5;
phi = p.Phi(p, gamma);

alg_opts.optimize_x = true;

alphas = alpha*[1,1/4,1/16,1/64];

u_opt = cell(length(alphas),1);
alg_out = cell(length(alphas),1);
Nnodes_phi = cell(length(alphas),1);
l2_err_phi = cell(length(alphas),1);

uinit = p.u_zero;

for n = 1:length(alphas)
  alpha = alphas(n);
  
  alg_opts.u0 = uinit;
  [u_opt{n}, alg_out{n}] = PDAPmultisemidiscrete(p, y_d, alpha, phi, alg_opts);

  uinit = u_opt{n};
  
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

