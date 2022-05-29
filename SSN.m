function u = SSN(p, Sred, ref, alpha, phi, u0, NQ)
%{ 
%% Solve the problem: 
            min_u (1/2)*|Sred*u - ref|^2 + alpha*phi(|u|_1,2)
with a (damped) semismooth Newton method 

Discription:

Input: 
  p - the model
  Sred - current value of the network and the relues
  ref - output data
  alpha - 
  phi - penalty function:  phi here the non-convex part is writen as z+phi(z)
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
Id = eye(n*NQ);
obj = @(u) F.F(Sred*u - ref) + alpha*sum(phi.phi(computeNorm(u, NQ)));

% ssn constant
c = 1 + alpha*gamma;

% nonconvex part (only for scalar sparsity)
Dphima = @(u)  (phi.dphi(abs(u))-1).*sign(u);
DDphima = @(u) phi.ddphi(abs(u));

% prox stuff
Pc = @(q) computeProx(q, alpha/c, NQ);
G = @(q, u) c*(q - u) + alpha*Dphima(u) + Sred'*F.dF(Sred*u - ref);

%% initial q
gf0 = alpha*Dphima(u0) + Sred'*F.dF(Sred*u0 - ref);
gf0 = reshape(gf0, NQ, n);
u0 = reshape(u0, NQ, n);
nu0 = sqrt(sum(abs(u0).^2, 1));
gf0(:,(nu0 > 0)) = bsxfun(@rdivide, -alpha*u0(:,(nu0 > 0)), nu0(nu0 > 0));
q = reshape(u0 - 1/c*gf0, NQ*n, 1);

% it should hold Pc(q) == u0
%disp(norm(Pc(q) - reshape(u0, NQ*n, 1)));

%% initialize SSN
u = Pc(q);
j = obj(u);
Gq = G(q, u);

j0 = j;
normGQ0 = norm(Gq);

iter = 0;
iterls = 0;
converged = false;

while ~converged && iter < 1666 && iterls < 30000
  
    DPc = computeDProx(q, alpha/c, NQ);
    DG = c*(Id - DPc) + alpha*diag(DDphima(u))*DPc + Sred'*F.ddF(Sred*u - ref)*Sred*DPc;

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

    %% Semismooth Newton update
    theta0 = 1/(1e-12*norm(DG,inf));

    %if condest(DG) > 1e15
    %    keyboard
    %end
    
    dq = - (DG + 1/theta0*Id)\Gq;
    qnew = q + dq;
    unew = Pc(qnew);
    jnew = obj(unew);

    %% Possible damping
    theta0 = max(norm(dq)/norm(Gq), theta0);
    theta = theta0;
    while isnan(jnew) || jnew > (1+1000*eps)*j
        % require descent only up to 10 times machine precision
        % for (jnew > j), the line-search does not terminate, occasionally
 
        %% damped Newton update
        qnew = q - (DG + 1/theta*Id)\Gq;

        unew = Pc(qnew);
        jnew = obj(unew);
        theta = theta/4;
        iterls = iterls + 1;
    end
    q = qnew;
    u = unew;
    jold = j;
    j = jnew;
    
    Gq = G(q, u);

    %fprintf('\t\tssn: %i, supp: %i, desc: %1.1e, res: %1.1e, damp: %1.1e\n', ...
    %       iter, nnz(computeNorm(u, NQ)), jold-j, norm(Gq), theta/theta0);
    iter = iter + 1;

    converged = ((theta == theta0) && (norm(Gq) < max(1e-12, 1e-10*normGQ0)));

end
fprintf('\tssn: %i, supp: %i, desc: %1.1e, res: %1.1e, damp: %1.1e\n', ...
        iter, nnz(computeNorm(u, NQ)), j0-j, norm(Gq), theta/theta0);

%fprintf('\tssn iter %i, damping: %i\n', iter, iterls);

end
