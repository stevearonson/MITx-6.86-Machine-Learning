import numpy as np
import kmeans
import common
import naive_em
import em
import pandas as pd

X = np.loadtxt("toy_data.txt")

'''
Question 2 - K-means
'''

temp_res = []
for K in [1,2,3,4]:
    for seed in [0,1,2,3,4]:
        mixture, post = common.init(X, K, seed)
        #common.plot(X, mixture, post, 'Init')
        
        mixture, post, cost = kmeans.run(X, mixture, post)
        #common.plot(X, mixture, post, 'Final with cost ' + str(cost))
        bic = common.bic(X, mixture, cost)

        temp_res.append(['K-means', K, seed, cost, mixture, post, bic])
        
results1 = pd.DataFrame(temp_res, columns=['Model', 'K', 'seed', 'cost', 'mixture', 'post', 'bic'])
print(results1.groupby('K')['cost'].min())

#for row in results1.loc[results1.groupby('K')['cost'].idxmin()].itertuples():
#        title = 'K = ' + str(row.K) + ' with cost = ' + str(cost)
#        common.plot(X, row.mixture, row.post, title)

'''
Question 3 - EM
'''

temp_res = []
for K in [1,2,3,4]:
    for seed in [0,1,2,3,4]:
        mixture, post = common.init(X, K, seed)
        #common.plot(X, mixture, post, 'Init')
        
        mixture, post, cost = naive_em.run(X, mixture, post)
        #common.plot(X, mixture, post, 'Final with cost ' + str(cost))
        
        bic = common.bic(X, mixture, cost)

        temp_res.append(['EM', K, seed, cost, mixture, post, bic])
        
results2 = pd.DataFrame(temp_res, columns=['Model', 'K', 'seed', 'cost', 'mixture', 'post', 'bic'])
print(results2.groupby('K')['cost'].min())

#for row in results2.loc[results2.groupby('K')['cost'].idxmax()].itertuples():
#        title = 'K = ' + str(row.K) + ' with cost = ' + str(cost)
#        common.plot(X, row.mixture, row.post, title)

'''
Question 8 - Netflix
'''
X = np.loadtxt('netflix_incomplete.txt')
X_gold = np.loadtxt('netflix_complete.txt')

temp_res = []
for K in [1,12]:
    for seed in [0,1,2,3,4]:
        mixture, post = common.init(X, K, seed)
        #common.plot(X, mixture, post, 'Init')
        
        mixture, post, cost = em.run(X, mixture, post)
        #common.plot(X, mixture, post, 'Final with cost ' + str(cost))
        
        bic = common.bic(X, mixture, cost)

        temp_res.append(['EM', K, seed, cost, mixture, post, bic])
        
netflix = pd.DataFrame(temp_res, columns=['Model', 'K', 'seed', 'cost', 'mixture', 'post', 'bic'])

print('Netflix Model Results:')
print(netflix.groupby('K')['cost'].max())

for row in netflix.loc[netflix.groupby('K')['cost'].idxmax()].itertuples():
    X_fill = em.fill_matrix(X, row.mixture)
    print('K = %d, RMSE = %f' % (row.K, common.rmse(X_gold, X_fill)))
