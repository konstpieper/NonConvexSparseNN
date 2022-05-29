function DP = computeDProx(v, mu, N)

%% the derivative of prox can not be represented as a complex number
assert(isreal(v));

%% normalize input and compute vector-norms
n = numel(v)/N;
v = reshape(v, N, n);

normsv = sqrt(sum(abs(v).^2, 1));

%% safeguard against division by zero
normsv_safe = max(normsv, (mu+eps)*eps);

%% for positive measures
%normsv(v <= 0) = eps;

%% compute non-smooth derivative of prox-operator
DP = zeros(N*n);

for cni = 1:n
   ind = (cni-1)*N+1:cni*N;
   DP(ind,ind) = max(0, 1 - mu / normsv_safe(cni)) * eye(N) ... 
       + (normsv(cni) >= mu) * mu / (normsv_safe(cni)^3) * v(:,cni)*v(:,cni)';
end

end
