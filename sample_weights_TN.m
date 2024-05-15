function [a_cand, b_cand] = sample_weights_TN(xhat, radius, N_target)

  dim = size(xhat, 1);

  %% random unit vectors
  a_cand = randn(dim, N_target);
  a_cand = a_cand ./ sqrt(sumsq(a_cand, 1));

  %% random loc in [-1,1]^n
  b_cand = (2 * rand(1, N_target) - 1) * radius;
  
end
