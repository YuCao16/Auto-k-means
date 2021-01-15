# This is a sample Python script.

# Press ⇧F10 to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import softmax

class Kmeanpp():

    def __init__(self, dataset, maxCluster=2, ifplot=False):  # dataset should be a nparray
        self.dataset = dataset
        self.maxCluster = maxCluster
        self.Centers = np.array([[]])
        self.numRow, self.numCol = self.dataset.shape
        self.oldNewArr = np.zeros((2, len(self.dataset)))
        self.oldNewArrSwitch = 0  # control the position of new arrange point will be store in self.oldNewArr
        self.uselsee = 1
        self.maxK = range(2, max(int(self.numRow / 20) + 1, 4), 1)

    def selectCenter(self, c=np.array([1])):
        if len(self.Centers[0]) == 0:
            k = int(np.random.randint(len(self.dataset), size=1))
            self.Centers = np.array([self.dataset[k]])
            return self.Centers
        else:
            a = np.random.uniform(c[0],c[-1])
            for i in range(len(c)):
                if i == 0:
                    if a <= c[i]:
                        self.Centers = np.vstack((self.Centers, self.dataset[i]))
                        return self.Centers
                else:

                    if c[i - 1] <= a <= c[i]:
                        self.Centers = np.vstack((self.Centers, self.dataset[i]))
                        return self.Centers
                    else:
                        self.useless = c

    def closestDistance(self, Centers, inside=True):  # return a matrix with element of the distance
        a = np.zeros([len(Centers), len(self.dataset)])

        for i in range(len(Centers)):
            for j in range(self.numRow):
                nd = 0
                for k in range(self.numCol):
                    nd = nd + (Centers[i, k] - self.dataset[j, k]) ** self.numCol
                a[i, j] = np.abs(nd)

        if inside:
            b = np.zeros(self.numRow)
            for i in range(self.numRow):
                b[i] = min(a[:, i])
            return b
        else:
            return a

    def calP(self, b):  # input a nparray b, b.shape is equal to the original self.dataset
        sumDist = np.sum(b)
        c = np.array([i / sumDist for i in b])
        c = np.cumsum(c)
        return c  # output c has same shape as b

    def getIniCenters(self):
        for i in range(self.maxCluster):
            if i == 0:
                self.selectCenter()
            else:
                add = self.closestDistance(Centers=self.Centers)
                prop = self.calP(add)
                self.selectCenter(prop)
        return self.Centers

    def arrangedPoint(self, existIni=True):
        if existIni:
            Centers = self.Centers
        else:
            Centers = self.getIniCenters()
        a = self.closestDistance(Centers, inside=False)
        b = np.zeros((self.numRow, 1))
        for i in range(self.numRow):
            minimal = np.min(a[:, i])
            for j in range(Centers.shape[0]):
                if a[j, i] == minimal:
                    b[i] = j
                    continue
        b = b.reshape(-1)
        b = b.astype('int')
        if self.oldNewArrSwitch == 0:
            self.oldNewArr[0] = b
            self.oldNewArrSwitch = 1
        elif self.oldNewArrSwitch == 1:
            self.oldNewArr[1] = b
            self.oldNewArrSwitch = 2
        else:
            self.oldNewArr[0] = b
            self.oldNewArrSwitch = 1

        return b

    def improvement(self, ifplot=False):

        b = self.getLatestArr()
        a = np.hstack((b.reshape((-1, 1)), self.dataset))
        b = [[]] * self.maxCluster
        newCenter = np.zeros((len(self.Centers), self.numCol))

        for i in range(len(self.Centers)):
            b[i] = [j[1:] for j in a if j[0] == i]

        if ifplot:
            return b

        for i in range(len(b)):
            j = np.array(b[i])
            if (len(j))==0:
                print('stop')
            for k in range(self.numCol):
                newCenter[i, k] = np.mean(j[:,k])
        self.Centers = newCenter
        self.arrangedPoint()
        self.uselsee = b
        return newCenter

    def checkIfCon(self):  # check if the new arrangement is equal to the previous one
        return (self.oldNewArr[0] == self.oldNewArr[1]).all()

    def getLatestArr(self):  # get the lastest arranged point
        if self.oldNewArrSwitch == 1:
            return self.oldNewArr[0]
        elif self.oldNewArrSwitch == 2:
            return self.oldNewArr[1]
        else:
            return None

    def k_mean(self, needPlot=False, maxCluster=2):
        self.maxCluster = maxCluster
        initialGuess = self.getIniCenters()
        self.arrangedPoint()
        initialArr = self.getLatestArr()
        kk = 0
        while kk < 10000:
            finalCenter = self.improvement()
            if self.checkIfCon():
                self.arrangedPoint()
                finalCenter = self.improvement()
                Loss = np.sum(self.closestDistance(finalCenter))
                if needPlot:
                    self.plot(finalCenter)
                return finalCenter, Loss
                break
            kk = kk + 1

    def process(self, needArr=False, needPlot=False):
        storeLoss = np.array([])
        kopt = 2
        for i in self.maxK:
            self.Centers = np.array([[]])
            self.maxCluster = i
            self.oldNewArr = np.zeros((2, len(self.dataset)))
            self.oldNewArrSwitch = 0
            Centers, Loss = self.k_mean(maxCluster=self.maxCluster)
            storeLoss = np.append(storeLoss, Loss)
            if self.ifKopt(storeLoss):
                kopt = len(storeLoss) + 1
                Centers = self.Centers
                Loss = storeLoss[-2]
                finalArr = self.getLatestArr()
                break
        if needArr:
            return Centers, kopt, Loss, finalArr
        if needPlot:
            self.plot(Centers)
        return Centers, kopt, Loss

        pass

    def ifKopt(self, storeLoss):  # check if we got a optimal value-k according to Loss
        if len(storeLoss) < 2:
            return False
        else:
            if storeLoss[-1] / storeLoss[-2] > 0.8 and storeLoss[-1]/ storeLoss[0] < 0.5:
                return True

    def plot(self, Centers):
        plotReq = self.improvement(ifplot=True)
        plt.figure()
        for i in range(len(Centers)):
            arr = np.array(plotReq[i])
            cen = np.array(Centers[i])
            plt.scatter(arr[:, 0], arr[:, 1])
            plt.scatter(cen[0], cen[1])
        plt.show()
        pass


dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values
print(len(X))
a = Kmeanpp(X)
# b = a.arrangedPoint()
# c, d= a.k_mean(needPlot=True, maxCluster=5)
c,d,e = a.process(needPlot=True)
print(c)
print("----------")
print(d)
print("----------")
print(e)
# k = 0
# for i in range(100):
#     a = Kmeanpp(X)
#     a,b,c = a.k_mean(maxCluster=5)
#     if b > 50000:
#         k+=1
#         # print(a)

# print(k)
# print(f)
# X = np.array([X[:, 0], X[:, 1]]).reshape(2, -1)
# plt.scatter(X[0], X[1])
# plt.show()
# a1 = np.array(e[0])
# a2 = np.array(e[1])
# a3 = np.array(e[2])
# a4 = np.array(e[3])
# a5 = np.array(e[4])
# c,d,e = a.k_mean()
# plt.figure()
# plt.scatter(a1[:, 0], a1[:, 1], s=100, c='red', label='Cluster 1')
# plt.scatter(a2[:, 0], a2[:, 1], s=100, c='blue', label='Cluster 2')
# plt.scatter(a3[:, 0], a3[:, 1], s=100, c='green', label='Cluster 3')
# plt.scatter(a4[:, 0], a4[:, 1], s=100, c='cyan', label='Cluster 4')
# plt.scatter(a5[:,0], a5[:,1], s = 100, c = 'magenta', label = 'Cluster 5')
# plt.scatter(c[:, 0], c[:, 1], s=300, c='yellow', label='Centroids')
# plt.title('Clusters of customers')
# plt.xlabel('Annual Income (k$)')
# plt.ylabel('Spending Score (1-100)')
# plt.legend()
# plt.show()

# dataset = np.array([[1, 2, 2], [1, 1, 2], [2, 1, 2], [2, 2, 2],
#                     [5, 5, 5], [6, 6, 5], [5, 6, 5], [6, 5, 5],
#                     [15, 15, 15], [16, 16, 15], [15, 16, 15], [16, 15, 15]])
# maxCluster = 3
# a = Kmeanpp(dataset, maxCluster)
# # b = a.getCenters()
# # print(b)
# c= a.arrangePoing()
# print(c)
# # for i in range(20):
# #     a = Kmeanpp(dataset, maxCluster)
# #     b = a.getCenters()
