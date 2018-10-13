function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

z = -1*z;
o = ones(size(z));
g = o./ (1+ power(e, z));
end
