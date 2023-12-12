function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
% nnCostFunction implements the neural network cost function for a three layer
% neural network which performs classification
%   [J grad] = nnCostFunction(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the theta matrices. 
%   
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.


% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our neural network

Theta1 = reshape(nn_params(1:(hidden_layer_size * (input_layer_size + 1))), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
dlji1 = zeros(size(Theta1));
dlji2 = zeros(size(Theta2));
c = size(dlji1);
zer=zeros(1,c(1,2));
dlji1 = [zer; dlji1];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Perform forward propagation through the the neural network 
%         and return the cost in the variable J. You can verify that your
%         cost function computation is correct by verifying the cost
%         computed in PA5.m Part 2. I suggest compute the cost *without* 
%         regularization  first so that it will be easier for you to debug.
%         You can then implement the regularized cost and check that in 
%         PA5.m Part 3. 
row = ones(m,1);
X = [row X]; % Add ones to the X data matrix
A1=X;
z1 = A1 * Theta1';
a1 = sigmoid(z1);
a1 = [row a1];
z2 = a1 * Theta2';
a2 = sigmoid(z2);
h = zeros(m,num_labels);

for i = 1:length(a2)
    h(i,y(i,1)) = 1;
end

J_unreg = -(1/m) * sum(sum(((h.*log(a2)) + (1-h).*log(1-a2))));
J = J_unreg + (lambda/(2*m))*(sum(sum((Theta1(:,2:end)).^2)) + sum(sum((Theta2(:,2:end)).^2)));
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. You can check that your implementation is 
%         correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               K dimentional binary vector of 1's and 0's to be used with the 
%               neural network cost function.
%
%         Hint: I recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
%




for t = 1:m
del3 = a2(t,:) - h(t,:);
del2 = (del3*Theta2).*a1(t,:).*(1-a1(t,:));

dlji1 = dlji1 + (del2' * A1(t,:));
dlji2 = dlji2 + (del3' * a1(t,:));

end

d_lji1 = (1/m) .* (dlji1(2:end,:));
d_lji1(:,2:end) = d_lji1(:,2:end) + ((1/m).*(lambda.*Theta1(:,2:end)));

d_lji2 = (1/m).*(dlji2);
d_lji2(:,2:end) = d_lji2(:,2:end) + ((1/m).*(lambda.*Theta2(:,2:end)));
Theta1_grad=d_lji1;
Theta2_grad=d_lji2;














% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
