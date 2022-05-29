function u = SSN_TR(p, Sred, ref, alpha, phi, u0, NQ)
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


%% vector case is not working right now...
assert(NQ == 1)
  
n = size(u0, 1)/NQ;

F = p.obj;
gamma = phi.gamma;
    
% some operators
Id = eye(n*NQ);
obj = @(u) F.F(Sred*u - ref) + alpha*sum(phi.phi(computeNorm(u, NQ)));

%% ssn constant
%c = 1 + alpha*gamma
c = alpha*(1+gamma);

% nonconvex part (only for scalar sparsity, NQ==1)
Dphima = @(u)  (phi.dphi(abs(u)) - 1) .* sign(u);
DDphima = @(u)  phi.ddphi(abs(u));

% prox stuff
Pc = @(q) computeProx(q, alpha/c, NQ);
G = @(q, u) c*(q - u) + alpha*Dphima(u) + Sred'*F.dF(Sred*u - ref);

%% initial q
gf0 = alpha*Dphima(u0) + Sred'*F.dF(Sred*u0 - ref);
gf0 = reshape(gf0, NQ, n);
u0 = reshape(u0, NQ, n);
nu0 = sqrt(sum(abs(u0).^2, 1));
gf0(:,(nu0 > 0)) = bsxfun(@rdivide, -alpha*u0(:,(nu0 > 0)), nu0(nu0 > 0));
ngf0 = sqrt(sum(abs(gf0).^2, 1));
q0 = u0 - 1/c*gf0;
q0(:,(nu0 == 0)) = (1/c) * bsxfun(@rdivide, -alpha * gf0(:,(nu0 == 0)), max(ngf0(nu0 == 0), alpha));

q0 = reshape(q0, NQ*n, 1);
% it should hold Pc(q0) == u0
err = norm(Pc(q0) - reshape(u0, NQ*n, 1));

if err >= 1e-9
  keyboard
end

%% initialize SSN
q = q0;
u = Pc(q);
j = obj(u);
r = G(q, u);

j0 = j;
normr0 = norm(r);

%% remove zeros immediately
experimental = true

%% trust region parameters
sigmamax = 100;
sigma = 0.01 * sigmamax;
maxiter = 6666;

%% semismooth Newton iteration
iter = 1;
dgnst = zeros(0, 4);

while (iter < maxiter)

  %figure(99);
  %plot([q,u])
  %pause(1);
  
    DPc = computeDProx(q, alpha/c, NQ);
    I = (diag(DPc) ~= 0);
    
    %disp(I')

    %% check for convergence
    rr = r' * r;                
    update =  (sqrt(rr) > max(1e-8, 1e-4*normr0)) || iter == 1;
 
    fprintf('Iteration %d: j = %e, |I| = %d, |g| = %1.3e\n', ...
            iter, j, sum(I), sqrt(rr));
    dgnst(iter,:) = [iter, j, sum(I), sqrt(rr)];

    if (~update)
      %keyboard;
      break;
    end

    DG = c*(Id - DPc) + alpha*diag(DDphima(u))*DPc + Sred'*F.ddF(Sred*u - ref)*Sred*DPc;
    
    %% compute Newton update
    kmaxit = 2*sum(I);
    [dq, flag, pred, relres, iter1] = mpcg(DG, -r, 1e-3, kmaxit, sigma, DPc);
    fprintf('\tKrylov: pred.desc: %1.3e relres: %1.1e, iter: %3d (%s)\t', pred, relres, iter1(end), flag); 

    %% apply update (preliminary)
    qnew = q + dq;
    
    %% try new state (preliminary)
    unew = Pc(qnew);
    jnew = obj(unew);

    %if abs(pred) <= eps
    %  jnew - j - 1e-10 * abs(j)
    %  keyboard;
    %end
    
    %% trust region computation
    sigmaold = sigma;

    if (isnan(jnew) || isinf(jnew) || jnew > j + 1e-10 * abs(j))
        %% reject and update radius
        sigma = .2 * sigma;
            
        fprintf('\treject: %1.3e > 0 \t %1.2e -> %1.2e\n', ...
                jnew - j, sigmaold, sigma);

        %disp(I')
        %disp(unew')

        %keyboard;
    else
        %% accept and update radius
        model = pred;
        %du = DPc(dq);
        %model = (du' * r + 0.5 * du' * DR(dq))
        rho = (jnew - j) / model;
            
        if (abs(rho - 1) < 0.2 || abs(j - jnew) / abs(j) < 1e-10)
            sigma = min(2 * sigma, sigmamax);
        elseif (abs(rho - 1) > 0.6)
            sigma = 0.4 * sigma;
        end
        fprintf('\trho = %f: \t %1.2e -> %1.2e\n', rho, sigmaold, sigma);
            
        %% apply update
        q = qnew;

        u = unew;
        j = jnew;
        
        %% new q
        gf = alpha*Dphima(u) + Sred'*F.dF(Sred*u - ref);
        %rrr = norm(c*(q-u) + gf);
        gf = reshape(gf, NQ, n);
        q = reshape(q, NQ, n);
        u = reshape(u, NQ, n);
        nu = sqrt(sum(abs(u).^2, 1));
        
        if experimental
          %% disable zero nodes (forever)
          zro = (nu == 0);
          Sred(:,zro) = 0;
          
          obj = @(u) F.F(Sred*u - ref) + alpha*sum(phi.phi(computeNorm(u, NQ)));
          G = @(q, u) c*(q - u) + alpha*Dphima(u) + Sred'*F.dF(Sred*u - ref);

          gf(:,zro) = 0;
        end
          
        ngf = sqrt(sum(abs(gf).^2, 1));
        q(:,(nu == 0)) = (1/c) * bsxfun(@rdivide, -alpha * gf(:,(nu == 0)), max(ngf(nu == 0), alpha));
        q = reshape(q, NQ*n, 1);
        u = reshape(u, NQ*n, 1);

        %% it should hold Pc(q) == u
        %disp(norm(Pc(q) - u));

        %% new residual
        r = G(q, u);

        %disp([rrr, norm(r)]);
        %II = ( (nu == 0) & (ngf >= alpha) );
        %disp(II);
    end
    iter = iter+1;
end
            
if (iter == maxiter)
  fprintf('TR-SSN maxiter reached\n');
  %keyboard;
end


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


end
