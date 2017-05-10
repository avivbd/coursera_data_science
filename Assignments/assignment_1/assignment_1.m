function [] = assignment_1()
%%
rng(1985)
A = [ones(10,1), randn(10,3)];
b = rand(10,1);
xx = A\b

x_init= zeros(4,1);
alpha = 0.0001;
matrixInverseVector(A, b, x_init, alpha)

end


function x = matrixInverseVector(A, b, x_init, alpha)
    x = x_init;
    while 1
      xnew = x - alpha*2*A'*(A*x-b)/length(b);
      tol = abs(norm(A*xnew-b) - norm(A*x-b))/norm(A*x-b)*100;
      if tol<0.0000001
        break
      end
      x = xnew;
    end  
end