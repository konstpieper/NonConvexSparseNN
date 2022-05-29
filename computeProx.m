function vprox = computeProx(v, mu, N)

%% normalize input and compute vector-norms
[n1, n2] = size(v);
n = numel(v) / N;
v = reshape(v, N, n);
normsv = sqrt(sum(abs(v).^2, 1));

%% safeguard against division by zero
normsv_safe = max(normsv, (mu+eps)*eps);

%% for positive measures
%normsv(v <= 0) = eps; %% eps: machine precision

%% apply soft shrinkage operator
vprox = bsxfun(@times, max(0, 1 - mu ./ normsv_safe), v);

%% reshape to original dimensions
vprox = reshape(vprox, n1, n2);

end
