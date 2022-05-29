function u = ProxGradPhi(p, Sred, ref, alpha, phi, u0, NQ)
%{ 
%% Solve the problem: 
            min_u (1/2)*|Sred*u - ref|^2 + alpha*phi(|u|_1,2)
with a proximal Newton method 

Discription:

Input: 
  p - the model
  Sred - current value of the network and the relues
  ref - output data
  alpha - 
  phi - penalty function
  u0 - current coefficients
  NQ - the target function output dimension 
  
Output:
  u -  new network
  
Other functions and parematers:
  obj -  the objective function
  F - computes norms of an array  
  Pc - zeros out terms in the array that are small
%}

n = size(u0, 1)/NQ;

F = p.obj;
gamma = phi.gamma;
    
% some operators
obj = @(u) F.F(Sred*u - ref) + alpha*sum(phi.phi(computeNorm(u, NQ)));
Phi = @(u) alpha*sum(phi.phi(computeNorm(u, NQ)));

% prox stuff
P = @(sigma, u) sign(u) .* phi.prox(sigma*alpha, abs(u));
G = @(u) Sred'*F.dF(Sred*u - ref);

L = max(eig(Sred'*F.ddF(u0)*Sred));

tau = 1/10;  % in (0, 1] (1 is classical prox gradient)

%% initialize
u = u0;
j = obj(u);
Gu = G(u);

j0 = j;
sigma = 1/L;
normG0 = norm(P(sigma, u - sigma*Gu) - u)/sigma;

Ghat = Gu;
weight = 1;
%v = u0;

iter = 0;
iterls = 0;
converged = false;

while ~converged && iter < 66666 && iterls < 150000

    %thk = 2/(iter + 2);
    %y = (1 - thk) * u + thk * v;
    
    %unew = P(sigma, y - sigma * G(y));
    
    %v = u + (1/thk) * (unew - u);

    unew = P(sigma, u - sigma*Ghat);
    jpred = j + (unew - u)'*Gu + 1/(2*sigma)*norm(unew - u)^2 - Phi(u) + Phi(unew);
    jnew = obj(unew);
    
    %% consistency check
%     dq = rand(N*n, 1);
%     for cni = 7:10
%         tau = sqrt(10)^(-cni);
%         qp = q + tau*dq;
%         qm = q - tau*dq;
%         fprintf('diff: %d\n', ...
%                  dq'*( DPc*dq - (Pc(qp) - Pc(qm))/(2*tau) ));
%         %fprintf('diff2: %d, %d\n', ...
%         %        (Sred'*(Sred*du))'*du, 1/2*(norm(Sred*up - ref)^2 - 2*norm(Sred*u - ref)^2 + norm(Sred*um - ref)^2));
%     end
    
    %% Possible damping
    while isnan(jnew) || jnew > (1-10*eps)*jpred
        % require descent only up to 10 times machine precision
        % for (jnew > j), the line-search does not terminate, occasionally
 
        sigma = sigma / 1.5;
        
        unew = P(sigma, u - sigma*Ghat);
        jpred = j + (unew - u)'*Gu + 1/(2*sigma)*norm(unew - u)^2 - Phi(u) + Phi(unew);
        jnew = obj(unew);
        
        iterls = iterls + 1;
    end
    sigma = min(sigma * 1.1, 1/(eps + alpha*gamma));
    
    desc = j - jnew;
    
    u = unew;
    j = jnew;
    
    Gu = G(u);
    normG = norm(P(sigma, u - sigma*Gu) - u)/sigma;
    
    Ghat = (1-1/weight)*Ghat + (1/weight)*Gu;
    weight = 1 + weight*(1-tau);

    if mod(iter, 100) == 0
      fprintf('\t\tprox: %i, supp: %i, desc: %1.1e, res: %1.1e, stepsize: %1.1e\n', ...
            iter, nnz(computeNorm(u, NQ)), desc, normG, sigma*L);
    end 
    iter = iter + 1;

    converged = (normG < max(1e-7, max(1e-1*alpha, 1e-5*normG0)));

end
fprintf('\tprox: %i, supp: %i, desc: %1.1e, res: %1.1e, stepsize: %1.1e\n', ...
        iter, nnz(computeNorm(u, NQ)), j0-j, normG, sigma*L);

%fprintf('\tssn iter %i, damping: %i\n', iter, iterls);

end
