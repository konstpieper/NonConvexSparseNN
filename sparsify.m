function u = sparsify(u, K, NQ)
%{
Does the caratheodory thing
%} 
norms = sqrt(sum(abs(u.u).^2, 1));
[~, supp, norms] = find(norms); 

ured = u.u(:,supp);
% repeats the support NQ times and adds 
% NQ 
bigsupp = kron(ones(size(supp)),(1:NQ)) + kron((supp-1)*NQ, ones(1,NQ)); 
Kred = K(:,bigsupp);

uprime = bsxfun(@rdivide, ured, norms);

Images = Kred*bsxfun(@times, kron(eye(length(supp)), ones(NQ,1)), uprime(:));

N = null(Images);

if (size(N, 2) >= 1)
    lambda = N(:,1)';
    if sum(lambda) <= 0
        lambda = -lambda;
    end
    [tau, ind] = max(lambda./norms);
    u.u(:,supp) = bsxfun(@times, ured, (1 - lambda./norms/tau));
    u.u(:,supp(ind)) = [];
    u.x(:,supp(ind)) = [];
end

end

