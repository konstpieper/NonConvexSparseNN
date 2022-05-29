function [x, flag, pred, relres, iter] = mpcg(H, b, rtol, maxit, sigma, DP)

% projected inner product for the pcg method
%in = @(Mv, w) (Mv' * DP(w));

% inner product for x for radius calculation (full)
%inx = @(Mv, w) (Mv' * w);

%% initial guess
x = zeros(size(b));
  
%% solve r(x) = b - H(x) = 0
r = b;
z = r;
DPz = DP * z;
resres = r' * DP * z;

%% search direction
d = z;
DPd = DPz;

res0 = sqrt(resres); % residual (projected)
fres0 = sqrt(r' * r); % full residual
epseps = -0.; % current step decrease in the functional j(x) = (1/2) * DP(x)' * H(x) - DP(x)' * b
pred = -0.; % predicted overall decrease in the functional
%% NB: equal to -(1/2) decrease in the squared energy norm

iter = 0;
flag = 'convgd';

%fprintf('\tpcg: iter: %3i res: %f\n', iter, res0);

%while (sqrt(resres) >= res0*rtol)
while (sqrt(-epseps) >= sqrt(-pred)*rtol) && (sqrt(resres) >= res0 * (1e-2*rtol))

    iter = iter + 1;

    if(iter > maxit)
        flag = 'maxitr';
        iter = maxit;
        break;
    end
    
    Hd = H * d;
    
    gamma = Hd' * DPd;
    
    % negative curvature:
    if (gamma <= 0) 
        flag = 'negdef';
        %disp(resres)
        if sigma > 0
            [x, tau] = to_boundary(x, d, sigma);
            pred = pred - tau * resres + (1/2) * tau^2 * gamma;
            r = r - tau * Hd;
            resres = r' * (DP * r);
        end
        relres = sqrt(resres) / res0;
        
        %fprintf('\tpcg: neg.def.: pred: %e |x|: %1.4e relres: %1.4e\n', pred, sigma, relres);
        return;
    end
    
    alpha = resres / gamma;
    
    xnew = x + alpha * d;

    % trust region radius reached
    normx = sqrt(xnew' * xnew);
    if (sigma > 0) && (normx > sigma)
        flag = 'radius';
        [x, tau] = to_boundary(x, d, sigma);
        normx = sigma;
        pred = pred - tau * resres + (1/2) * tau^2 * gamma;
        r = r - tau * Hd;
        resres = r' * (DP * r);
        relres = sqrt(resres) / res0;

        %fprintf('\tpcg: radius:   pred: %e |x|: %1.4e relres: %1.4e\n', pred, normx, relres);
        return;
    else
        x = xnew;
    end

    r = r - alpha * Hd;

    epseps = - (1/2) * alpha * resres;
    %% NB: epseps = - alpha resres + (1/2) * alpha^2 gamma = -(1/2) alpha resres
    pred = pred + epseps;
    
    z = r;
    DPz = DP * z;

    resresold = resres;
    resres = r' * DPz;

    beta = - alpha * Hd' * DPz / resresold;
    %beta = resres / resresold;

    d = z + beta * d;
    DPd = DPz + beta * DPd;

    %fprintf('\tpcg: iter: %3i pred: %e |x|: %1.4e res: %1.4e indicator: %1.4e\n', iter, pred, normx, sqrt(resres), sqrt(epseps/pred));
end

% minimize |H(x + \theta z) - b|^2_Mi = |r - \theta H(z)|^2_Mi
Hz = H * z;
MiHz = Hz;
theta = (r' * MiHz) / (Hz' * MiHz);
%% TODO divison by zero???

xnew = x + theta * z;

% trust region radius reached in final step
normx = sqrt(xnew' * xnew);
if (sigma > 0) && (normx > sigma)
    flag = 'radius';
    [x, theta] = to_boundary(x, z, sigma);
    normx = sigma;
else
    x = xnew;
end

pred = pred - theta * resres + (1/2) * theta^2 * Hz' * DPz;

%fprintf('\tpcg: last:     pred: %e |x|: %1.4e \ttheta = %d\n', pred, normx, theta);

r = r - theta * Hz;
z = r;

%DPz = DP(z);
%resres = r' * DPz;
%relres = sqrt(resres) / res0;

relres = sqrt(r' * z) / fres0;

end


function [x, tau] = to_boundary(x, d, s)

dd = d' * d;
xd = x' * d;
xx = x' * x;

det = xd*xd + dd * (s*s - xx);
tau = (s*s - xx) / (xd + sqrt(det));

x = x + tau * d;

end
