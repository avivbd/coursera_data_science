function [] = simple_gradient_descent()
%     rng(1985)
    A = [ones(10,1), randn(10,3)];
    b = rand(10,1);

    x_init= zeros(4,1);
    alpha = 0.01;
    [x, n_iter, rtols] = matrixInverseVector(A, b, x_init, alpha);

    disp('n iterations')
    disp(n_iter)
    disp('x = ')
    disp(x)
    disp('A\b = ')
    disp(A\b)
    
    plot(log10(rtols))
    xlabel('iterations')
    ylabel('log rtol')
end


function [x, n_iter, rtols] = matrixInverseVector(A, b, x_init, alpha)
    x = x_init;
    n_iter = 1;
    rtols = [];
    while true
        grad = 2*A'*(A*x-b);
        xn = x - alpha*grad;
        sse = @(x) norm(A*x-b);
        rtol = abs(sse(xn) - sse(x))/sse(x);
        if rtol<1e-12
            break
        else
            x = xn;
            n_iter = n_iter + 1;
            rtols(n_iter) = rtol;
        end
    end  
end