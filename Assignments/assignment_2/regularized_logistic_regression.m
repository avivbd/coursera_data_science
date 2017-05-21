function [  ] = regularized_logistic_regression(  )
%REGULARIZED_LOGISTIC_REGRESSION carries out regularized logistic
%regression calculation
%
clc
clear
close all

[X, y] = read_the_data();
plot_the_data(X, y)

X = polynomial_transformer(X);
lambda = 1;
[theta] = optim_fun(X, y, lambda);

plotDecisionBoundary(theta);
title(sprintf('lambda = %g', lambda))


end

function [X, y] = read_the_data()
filename = 'ex2data2.txt';
filepath = '../assignment_2/machine-learning-ex2/ex2/';
Xcols = [1,2]; 
ycols = 3;

data = load([filepath, filename]);
X = data(:, Xcols);
y = data(:, ycols);

end

function plot_the_data(X, y)
xlab = 'Microchip Test 1';
ylab = 'Microchip Test 2';
leg = {'y = 1', 'y = 0'};
plt1 = 'k+';
plt0 = 'ko';
plt0_mfc = 'y';
plt0_ms = 7;

hold on;
hp1 = plot(X(y==1,1), X(y==1,2), plt1);
hp0 = plot(X(y==0,1), X(y==0,2), plt0);
hp0.MarkerFaceColor = plt0_mfc;
hp0.MarkerSize = plt0_ms;
xlabel(xlab)
ylabel(ylab)
legend(leg)


end

function [theta] = optim_fun(X, y, lambda)
initial_theta = zeros(size(X, 2), 1);

options = optimset('GradObj', 'on', 'MaxIter', 400, ...
    'Algorithm', 'trust-region');
[theta, ~, ~] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);

end

function [J, grad_J] = costFunctionReg(theta, X, y, lambda)
m = length(y); 
L = eye(length(theta));
L(1) = 0;
h = sigmoid(X*theta);
reg_term = 0.5*lambda*sum(theta(2:end).^2);
ls_term = y'*log(h) + (1-y')*(log(1 - h));
J = -1/(m) * (ls_term + reg_term );
grad_J = 1/m *(X'*(h - y) + lambda*L*theta);

end

function g = sigmoid(z)

g = 1./(1 + exp(-z));

end

function out = polynomial_transformer(X)
X1 = X(:,1);
X2 = X(:,2);
degree = 6;
out = ones(size(X1(:,1)));

for i = 1:degree
    for j = 0:i
        out(:, end+1) = (X1.^(i-j)).*(X2.^j);
    end
end

end

function plotDecisionBoundary(theta)
u = linspace(-1, 1.5, 50);
v = linspace(-1, 1.5, 50);

z = zeros(length(u), length(v));
% Evaluate z = theta*x over the grid
for i = 1:length(u)
    for j = 1:length(v)
        z(i,j) = polynomial_transformer([u(i), v(j)])*theta;
    end
end
z = z'; % important to transpose z before calling contour

% Plot z = 0
% Notice you need to specify the range [0, 0]
contour(u, v, z, [0, 0], 'LineWidth', 2)
    
end
