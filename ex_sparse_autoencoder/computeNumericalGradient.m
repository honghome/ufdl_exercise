function numgrad = computeNumericalGradient(J, theta)
% numgrad = computeNumericalGradient(J, theta)
% theta: a vector of parameters
% J: a function that outputs a real-number. Calling y = J(theta) will return the
% function value at theta. 
  
% Initialize numgrad with zeros
numgrad = zeros(size(theta));
%% ---------- YOUR CODE HERE --------------------------------------
% Instructions: 
% Implement numerical gradient checking, and return the result in numgrad.  
% (See Section 2.3 of the lecture notes.)
% You should write code so that numgrad(i) is (the numerical approximation to) the 
% partial derivative of J with respect to the i-th input argument, evaluated at theta.  
% I.e., numgrad(i) should be the (approximately) the partial derivative of J with 
% respect to theta(i).
%                
% Hint: You will probably want to compute the elements of numgrad one at a time. 

%perturb = zeros(size(theta));
% e = 1e-4;
% for p = 1:numel(theta)
%    perturb(p) = e;
%    loss1 = J(theta - perturb);
%    loss2 = J(theta + perturb);
%    numgrad(p) = (loss2 - loss1) / (2*e);
%    perturb(p) = 0;
% end
e = 1e-4;
thetaNum = numel(theta);
mat1 = repmat(theta, 1, thetaNum) - diag(repmat(e, thetaNum, 1));
mat2 = repmat(theta, 1, thetaNum) + diag(repmat(e, thetaNum, 1));
for i = 1: thetaNum
	numgrad(i) = (J(mat2(:,i)) - J(mat1(:,i)));
end
numgrad = numgrad ./ (2*e);

%% ---------------------------------------------------------------
end
