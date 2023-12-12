function p = NNpredict(Theta1, Theta2, X)
% NNpredict predicts the label of an input given a trained neural network
%   p = NNpredict(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained parameters of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: 1. you will use forward propagation to compute activation a2 and a3
%       2. The sigmoid function is defined in sigmoid.m. You can call it.
%       3. The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(a, [], 2) to obtain the max for each row.
%


row = ones(m,1);
X = [row X]; % Add ones to the X data matrix

z1 = X * Theta1';
a1 = sigmoid(z1);
a1 = [row a1];
z2 = a1 * Theta2';
a2 = sigmoid(z2);

[row col] = max(a2, [],2);


p = col;








    








% =========================================================================


end
