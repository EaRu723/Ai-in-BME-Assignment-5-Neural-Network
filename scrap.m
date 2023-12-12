%%

clear all
close all
X = rand(5000,400);
row = ones(5000,1);
Xn = zeros(5000,401);
Xn = [row X]; % Add ones to the X data matrix
a=Xn';
%%
%  To give you an idea of the network's output, you can also run
%  through the examples one at the a time to see what it is predicting.

%  Randomly permute examples
rp = randperm(m);

for i = 1:m
    % Display 
   fprintf('\nDisplaying Example Image\n');
    displayData(X(rp(i), :));

    pred = NNpredict(Theta1, Theta2, X(rp(i),:));
    fprintf('\nNeural Network Prediction: %d (digit %d)\n', pred, mod(pred, 10));
    
    % Pause with quit option
    s = input('Paused - press enter to continue, q to exit:','s');
    if s == 'q'
      break
    end
end



[row_0 col_0] = find(y==10);
y(row_0,1) = 0;
y = y+1;

p = p-1;
[row_0 col_0] = find(p==0);
p(row_0,1) = 10;





%% ========= 3: Implement Regularization ========================
%  Once your cost function implementation is correct, you should now
%  continue to implement the regularization with the cost.

fprintf('\nChecking Cost Function (w/ Regularization) ... \n')

% regularization parameter (set to 1 here).
lambda = 1;

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);

fprintf(['Cost at parameters (loaded from PA5theta): %f '...
         '\n(this value should be about 0.383770)\n'], J);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ========= 4: Implement Backpropagation =======================
%  Once your cost function is verified, you should proceed to implement the
%  backpropagation algorithm for the neural network. Addd code in 
%  nnCostFunction.m to return the partial derivatives.
%
fprintf('\nChecking Backpropagation... \n');

%  Check gradients without regularization by running checkNNGradients
lambda = 0;  
checkNNGradients(lambda); 

fprintf('\nProgram paused. Press enter to continue.\n');
pause;
%  Check gradients with regularization by running checkNNGradients
lambda = 3;  
checkNNGradients(lambda); 

%% ========== 2: Compute Cost (Forward propagation) ==============

% Weight regularization parameter (we set this to 0 here).
lambda = 0;
% add code in nnCostFunction.m to compute and return cost J 
[J, grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);

fprintf(['Cost at parameters (loaded from PA5theta): %f '...
         '\n(this value should be about 0.287629)\n'], J);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;
%% ========= 3: Implement Regularization ========================
%  Once your cost function implementation is correct, you should now
%  continue to implement the regularization with the cost.

fprintf('\nChecking Cost Function (w/ Regularization) ... \n')

% regularization parameter (set to 1 here).
lambda = 1;

[J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);

fprintf(['Cost at parameters (loaded from PA5theta): %f '...
         '\n(this value should be about 0.383770)\n'], J);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ========= 4: Implement Backpropagation =======================
%  Once your cost function is verified, you should proceed to implement the
%  backpropagation algorithm for the neural network. Addd code in 
%  nnCostFunction.m to return the partial derivatives.
%
fprintf('\nChecking Backpropagation... \n');

%  Check gradients without regularization by running checkNNGradients
lambda = 0;  
checkNNGradients(lambda); 

fprintf('\nProgram paused. Press enter to continue.\n');
pause;
%  Check gradients with regularization by running checkNNGradients
%lambda = 3;  
%checkNNGradients(lambda); 

