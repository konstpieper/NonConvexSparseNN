function norms = computeNorm(vec, NQ)

n = numel(vec)/NQ;

vec = reshape(vec, NQ, n);
norms = sqrt(sum(abs(vec).^2, 1));

end
