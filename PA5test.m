% AI in BME Class - Programming Assignment 5
% Neural Networks - Part II

%  ------------ Instructions --------------------------------------------------
%
%  This file contains code that helps you get started. 
%  You will need to complete the following functions in this exericse:
%     
%     
%     nnCostFunction.m
%
%  For this part of exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% =========== 1: Loading Data and Parameters =====================

% -- Load Training Data -------
load('PA5data1.mat');
m = size(X, 1);

% -- Loading Pameters -------- 
% pre-initialized neural network parameters are loaded for testing purpose 

fprintf('\nLoading Saved Neural Network Parameters ...\n')
% Load the weights into variables Theta1 and Theta2
load('PA5theta.mat');

% Unroll parameters 
nn_params = [Theta1(:) ; Theta2(:)];

input_layer_size = 400;  % Input layer size
hidden_layer_size = 25;  % Hidden layer size
num_labels = 10;         % Number of labels


%% ========== 2: Compute Cost (Forward propagation) ==============

% Weight regularization parameter (we set this to 0 here).
lambda = 0;
% add code in nnCostFunction.m to compute and return cost J 
J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
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




