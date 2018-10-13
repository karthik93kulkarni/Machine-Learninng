function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
 

%initialization
m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));


%%%%%%%%%%%%%%%%%HINT PROVIDED BY THE COURSE%%%%%%%%%%%%%%%%%%%%%%%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

tempTheta = theta;
tempTheta(1) = 0;

J = (-1 / m) * sum(y.*log(sigmoid(X * theta)) + (1 - y).*log(1 - sigmoid(X * theta))) + (lambda / (2 * m))*sum(tempTheta.^2);
temp = sigmoid (X * theta);
error = temp - y;
grad = (1 / m) * (X' * error) + (lambda/m)*tempTheta; %Following the formula


grad = grad(:);

end
