import numpy as np
import scipy.linalg as la

def sdls(A,B,X0=None,Y0=None,tol=1e-10,verbose=False):
    """
     [X,Y,norm_res,muv,tt,iter,fail] = SDLS(A,B,X0,Y0,tol,verbose)

     Uses a stanadard path-following interior-point method based on the
     AHO search direction to solve the symmetric semidefinite constrained
     least squares problem:

       min  norm(A*X-B,’fro’)
       s.t. X symm. pos. semidef.

     where A and B are real m-by-n matrices, and X is a real n-by-n matrix.
     X0 and Y0 are n-by-n initial strictly feasible matrices, which means
     that X0 and Y0 are symmetric positive definite.
     Set as [] for the default value of eye(n).

     tol is the zero tolerance described below.
     Set as [] for the default value of 1e-10.

     Set verbose = 1 for screen output during algorithm execution,
     otherwise set vebose = 0 for no output.

     SDLS returns approximate optimal solutions to the above primal
     problem and its associated dual problem so that

       norm(res,’fro’)  <=  sqrt(tol)*norm(res0,’fro’)
            trace(X*Y)  <=  tol*trace(X0*Y0)

     where res = (Z+Z’)/2-Y, Z =  A’*(A*X-B), and res0 is res evaluated Appendix B.  Matlab M-files
     at X0, Y0.

     SDLS optionally returns:

      norm_res : norm(res,’fro’) at the final iterate,
      muv  : a vector of the duality gaps for each iteration
      tt   : the total running time of the algorithm
      iter : the number of iterations required
      fail : fail = 1 if the algorithm failed to achieve the desired
             tolerances within the maximum number of iterations allowed;
             otherwise fail = 0
     Nathan Krislock, University of British Columbia, 2003.

     N. Krislock. Numerical solution of semidefinite constrained least
     squares problems.  M.Sc. thesis, University of British Columbia,
     Vancouver, British Columbia, Canada, 2003.

     Translated to Python by Marcel Nonnenmacher, research institute caesar, 2016

    """

    max_iter = 100  # max iteration
    m,n = A.shape
    AA, AB, I = A.T.dot(A), A.T.dot(B), np.eye(n)
    X = np.eye(n) if X0 is None else X0.copy()
    Y = np.eye(n) if Y0 is None else X0.copy()

    XAA,XY = X.dot(AA), X.dot(Y);
    Z = XAA.T - AB  
    Z = (Z+Z.T)/2  
    R = Z - Y
    norm_res = np.linalg.norm(R)
    mu = np.trace(XY)/n  
    muv = np.zeros(max_iter)
    muv[0] = mu
    tol1 = np.sqrt(tol)*norm_res 
    tol2 = tol*mu
    r, theta = 0, 0 
    while ( norm_res > tol1 or mu > tol2 ) and r < max_iter : 
        # Compute sigma and tau
        sigma = 1/n**2 if norm_res < tol1 else 1-1/n**2
        tau = sigma*mu;
        # Compute the AHO search direction (dX,dY)
        E = (np.kron(I,Y)+np.kron(Y,I))/2
        XZ = X.dot(Z)
        M  = (np.kron(I,XAA)+np.kron(AA,X)+np.kron(X,AA)+np.kron(XAA,I))/4 + E
        d = (tau*I - (XZ+XZ.T)/2).reshape(-1,order='F')
        # d = F*vec(-R) + vec(tau*I-(X*Y+Y*X)/2);
        P,L,U = la.lu(M)

        #dx = U\(L\(P*d))        
        dx = np.linalg.solve(U, np.linalg.solve(L, d.dot(P)))
        
        dX = mat(dx)
        dX = (dX+dX.T)/2  
        AAdX = AA.dot(dX)
        dY = (AAdX+AAdX.T)/2 + R  
        dY = (dY+dY.T)/2;
        # Compute the step length theta
        c = 0.9 + 0.09*theta
        theta1, theta2 = max_step(X,dX), max_step(Y,dY)
        theta_max = np.min((theta1, theta2))
        theta = np.min((c*theta_max,1))
        # Update
        X += theta*dX
        Y += theta*dY
        XAA, XY = X.dot(AA), X.dot(Y)
        Z = XAA.T - AB  
        Z = (Z+Z.T)/2  
        R = Z - Y
        norm_res = np.linalg.norm(R)
        mu = np.trace(XY)/n  
        muv[r] = mu;
        r +=1
    fail = True if r==max_iter and ( norm_res > tol1 or mu > tol2 ) else False
    if fail and verbose:
        print('\n Failed to reach desired tolerance. \n')
        print(' (reached,set) tolerance is (norm_res, tol1) = ', (norm_res,tol1))
        print(' (reached,set) tolerance is (mu, tol2) = ', (mu, tol2))

    return X,Y,norm_res,muv,r,fail        
        
def mat(v,n=None):
    # V = mat(v,n)
    #
    # Given an m*n column vector v, returns the corresponding
    # m-by-n matrix V.  If n is not given, it is assumed that
    # v is an n^2 column vector.
    if n is None:
        n = int(np.sqrt(v.size))
        m = n  
        mn = m*n
    else:
        mn = v.size;
        m = mn//n;
    V, k = np.zeros((m,n)), 0
    for i in range(0,mn,m):
        V[:,k] = v[i:(i+m)]
        k += 1
    return V
                  
def max_step(X,dX):
    # theta = MAX_STEP(X,dX)
    #
    # Given n-by-n symmetric matrices, X and dX, where X is positive definite,
    # MAX_STEP returns the largest theta_max > 0 such that
    #
    #   X + theta*dX
    #
    # is positive definite for all 0 < theta < theta_max. If X + theta*dX is
    # positive definite for all theta, then theta_max = Inf.

    x = np.max(la.eig(-dX,X,right=False,left=False));
    
    if np.isfinite(x) and not np.allclose(x, np.real(x)):
        print('\n Warning: max_step returns complex argument')
        print('max_step: ', x)
        print('abs(max_step): ', np.abs(x))
        
    x = np.abs(x)
    return 1/x if x > 0 else np.inf
