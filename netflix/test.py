import numpy as np
import em
import common
import naive_em

#X = np.loadtxt("test_incomplete.txt")
#X_gold = np.loadtxt("test_complete.txt")
X = np.loadtxt('test_incomplete.txt')
X_gold = np.loadtxt('test_complete.txt')

K = 4
n, d = X.shape
seed = 0

mixture, post = common.init(X, K, seed)
print(mixture.mu)
print(mixture.var)
print(mixture.p)

post, cost = em.estep(X, mixture)
mixture = em.mstep(X, post, mixture)

print('Post = ', end='')
print(post)
print('Cost = ', end='')
print(cost)
print('Mu = ', end='')
print(mixture.mu)
print('Var = ', end='')
print(mixture.var)
print('P = ', end='')
print(mixture.p)
#common.plot(X, mixture, post, 'Test')

mixture, post = common.init(X, K, seed)
mixture, post, cost = em.run(X, mixture, post)

print('Final Cost = ', end='')
print(cost)
print('Final Mu = ', end='')
print(mixture.mu)
print('Final Var = ', end='')
print(mixture.var)
print('Final P = ', end='')
print(mixture.p)


X_fill = em.fill_matrix(X, mixture)
print('Estimated X:')
print(X_fill)

print('RMSE = %f' % common.rmse(X_gold, X_fill))