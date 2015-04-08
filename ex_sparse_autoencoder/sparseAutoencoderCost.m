function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 

% Feed-forward to compute h
m = size(data)(2);
a1 = data; % visiblesize x m
z2 = W1 * a1; % hiddensize x visiblesize * visiblesize x m --> hiddensize x m
for i = 1:m
    z2(:,i) = z2(:,i) + b1; % hiddensize + hiddensize
end

a2 = sigmoid(z2); % hiddensize x m
z3 = W2 * a2; % visiblesize x hiddensize * hiddensize x m --> visiblesize x m
for i = 1:m
    z3(:,i) = z3(:,i) + b2; % visiblesize + visiablesize
end

h = sigmoid(z3); % visiblesize x m


% backpropagation to compute gradient
for i = 1:m
    cost += sum((h(:,i) - data(:,i)) .^ 2);
end
cost = 1/(2*m) * cost;

% add weight decay
cost = cost + lambda/2 * (sum(sum(W1 .^ 2)) + sum(sum(W2 .^ 2)));

% add KL divergence
averActivation = mean(a2, 2);
cost = cost + beta * KLdivergence(averActivation, sparsityParam);

d3 = (h - data) .* sigmoidGradient(z3); % visiblesize x m
%d2 = (W2)' * d3 .* sigmoidGradient(z2); % (visiblesize x hiddensize)' * visiblesize x m + hiddensize x m
d2 = (W2)' * d3; % hiddensize x m
for i = 1:m
	d2(:,i) = d2(:,i) + beta * ((1-sparsityParam)./(1-averActivation) - sparsityParam./averActivation);
end
d2 = d2 .* sigmoidGradient(z2);

W1grad = d2 * a1' ./ m;	% hiddensize x visiblesize , d2(i,:) * a1'(:,j) is a accumulated value
W2grad = d3 * a2' ./ m;	% visiblesize x hiddensize
b1grad = mean(d2,2) ;	% hiddensize x m --> hiddensize x 1
b2grad = mean(d3,2);	% visiblesize x m --> visiblesize x 1

% add weight decay
W1grad = W1grad + lambda .* W1;
W2grad = W2grad + lambda .* W2;

%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

function g = sigmoidGradient(x)
    g = zeros(size(x));
    g = sigmoid(x) .* (1 - sigmoid(x));
end

function kl = KLdivergence(averActivation, sparsityParam)
	kl = sparsityParam .* log(sparsityParam ./ averActivation) + ...
	(1 - sparsityParam) .* log((1 - sparsityParam) ./ (1 - averActivation));
	kl = sum(kl);
end
