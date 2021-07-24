# CodeWrite 
#Write logistic regression code from scratch. Use gradient descent.
# Only write functions here
import numpy as np
import matplotlib.pyplot as plt
    
def kernel_matrix(A,B,kernel = 'linear',kernel_param=1.):
    Y = np.matmul(A,B.T) #n*m matrix
    if(kernel == 'linear'):
        return Y
    elif (kernel == 'poly'):
        return (1+Y[:][:])**kernel_param
    n = A.shape[0]
    m = B.shape[0]
    Z = np.matmul(A,A.T) #n*n matrix
    X1 = np.empty((n,m))
    for i in range(n):
        X1[i] = np.ones(m)*Z[i][i]
    Z = np.matmul(B,B.T) #m*m matrix
    X2 = np.empty((m,n))
    for i in range(m):
        X2[i] = np.ones(n)*Z[i][i]
    Z = X1 + X2.T  # n*m matrix
    return np.exp((2*Y-Z)*kernel_param)
    
def R_f(a):
    if(a<=-709):
        return (-1*a)
    return np.math.log(1+np.exp(-1*a))

vec_R = np.vectorize(R_f)
R = lambda alpha,Y,K,lamda : np.sum(vec_R(Y.T*(np.matmul(alpha.T,K))),axis = 0)+(lamda/2)*np.matmul(np.matmul(alpha,K).T,alpha)

def grad_f(a):
    if(a >= 709):
        return 0
    else :
        return (1/(1+np.exp(a)))
    
vec_grad = np.vectorize(grad_f)
    
grad = lambda alpha,Y,K,lamda :np.matmul(K,vec_grad(Y.T*(np.matmul(alpha.T,K))).T*Y)*(-1)+lamda*(np.matmul(K,alpha))
                                

def train_pred_logistic_regression(A, B, eeta,kernel='linear', reg_param=0.,kernel_param=1., num_iter_gd=100):
    
    X = np.copy(A)
    Y = np.copy(B)
    d = X[0].size
    n = Y.size
    Y.reshape(n,1)
    mu = np.sum(X,axis = 0)/n
    sigma = np.std(X,axis = 0)
    print(sigma)
    for i in range(n):
        X[i] = (X[i]-mu)/sigma
    K = kernel_matrix(X,X,kernel,kernel_param)
    alpha = np.zeros(n)
    alpha.reshape(alpha.shape[0],-1)
    g = -1
    A = []
    B = []
    out = np.copy(alpha)
    for i in range(num_iter_gd):
        gradient = grad(alpha,Y,K,reg_param)
        alpha = alpha - eeta*(gradient/np.linalg.norm(gradient))
        k = R(alpha,Y,K,reg_param)
        if(g == -1):
            g = k
        else:
           # if(g <= k):
           #     break
            g = k
        if (i > 0):
            A.append((i+1))
            B.append(g)
        out = np.copy(alpha)
            
    #plt.plot(A,B)
        
    return out

def output_predict(Y):
    return np.array([1 if(a>0) else -1 for a in Y])

def test_pred(alpha,A,B, kernel, kernel_param):
    
    train_X = np.copy(A)
    test_X = np.copy(B)
    n = train_X.shape[0]
    mu = np.sum(train_X,axis = 0)/n
    sigma = np.std(X_train,axis = 0)
    print(mu)
    print(sigma)
    for i in range(train_X.shape[0]):
        train_X[i] = (train_X[i]-mu)/sigma
    for i in range(test_X.shape[0]):
        test_X[i] = (test_X[i]-mu)/sigma
        
    K = kernel_matrix(train_X,test_X,kernel,kernel_param) # n*m matrix
    Y = np.matmul(alpha.T,K) #1*m matrix
    Y = output_predict(Y)
    Y.reshape(Y.shape[0],1)
    return Y

x = np.load("Data/dataset_A.npz")
#my_func(x)

X_train = x['arr_0']
Y_train = x['arr_1']
X_test =  x['arr_2']
Y_test =  x['arr_3']


alpha = train_pred_logistic_regression(X_train,Y_train,1e-2,'linear',0.001,0.1,300)
Y_pred = test_pred(alpha,X_train,X_test,'linear',0.1)
y_pred = test_pred(alpha,X_train,X_train,'linear',0.1)
c = 0
for i in range(Y_test.shape[0]):
     if (Y_test[i] != Y_pred[i]):
        c = c + 1
d = 0
for i in range(Y_train.shape[0]):
     if(Y_train[i] != y_pred[i]):
            d = d + 1
print("reg_param: ",0.001,",kern_param: ",0.1)
print("test accuracy: ",c/Y_test.shape[0])
print("train accuracy: ",d/Y_train.shape[0])
plt.show()
