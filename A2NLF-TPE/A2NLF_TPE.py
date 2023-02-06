# coding=utf-8
import numpy as np
from math import sqrt
import time
from time import perf_counter
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.early_stop import no_progress_loss
from numba import jit, objmode

dim = 20
max_evals = 100
early_stop = 20
delayCount = 5
datasetName = "Jester"
metric = "RMSE"
print("*******", datasetName, "*******")
print("******* Metric=", metric, "*******")
trainFile = "/Users/zhong/Desktop/codes/A2NLF-codes/A2NLF/DS/" + datasetName + "_train.txt"
valFile = "/Users/zhong/Desktop/codes/A2NLF-codes/A2NLF/DS/" + datasetName + "_val.txt"
testFile = "/Users/zhong/Desktop/codes/A2NLF-codes/A2NLF/DS/" + datasetName + "_test.txt"

def dataSetGenerator(separator, fileName):
    userMaxID = 0
    itemMaxID = 0
    dataSet = []
    with open(fileName, 'r') as f:
        for line in f.readlines():
            eachData = line.split(separator)
            userID = int(eachData[0])
            itemID = int(eachData[1])
            rating = float(eachData[2])
            userMaxID = userMaxID if (userID < userMaxID) else userID
            itemMaxID = itemMaxID if (itemID < itemMaxID) else itemID
            userID = userID - 1  # id从0开始
            itemID = itemID - 1
            dataSet.append((userID, itemID, rating))
    f.close()
    return dataSet, userMaxID, itemMaxID

def dataLoad(trainFileName, testFileName, separator):
    trainSet, trainUserMaxID, trainItemMaxID = dataSetGenerator(separator, trainFileName)
    testSet, testUserMaxID, testItemMaxID = dataSetGenerator(separator, testFileName)
    userMaxID = trainUserMaxID
    if testUserMaxID > trainUserMaxID : userMaxID = testUserMaxID
    itemMaxID = trainItemMaxID
    if testItemMaxID > trainItemMaxID : itemMaxID = testItemMaxID
    return trainSet, testSet, userMaxID, itemMaxID

def initRatingSetSize(trainSet, userMaxID, itemMaxID):
    userRSetSize = np.zeros(userMaxID)
    itemRSetSize = np.zeros(itemMaxID)
    for trainR in trainSet:
        userID = trainR[0]
        itemID = trainR[1]
        userRSetSize[userID] = userRSetSize[userID] + 1
        itemRSetSize[itemID] = itemRSetSize[itemID] + 1
    return userRSetSize, itemRSetSize

def initFeature_Random(userMaxID, itemMaxID, dim):
    print("准备生成随机初始值...")
    np.random.seed(0)
    P = np.random.uniform(0, 0.004, (userMaxID, dim))
    Q = np.random.uniform(0, 0.004, (itemMaxID, dim))
    print("随机初始值生成完毕！！！")
    return P, Q

@jit(nopython=True)
def tRMSE(P, Q, testSet):
    sumRMSE = 0
    sumCount = 0
    for testR in testSet:
        userID = int(testR[0])
        itemID = int(testR[1])
        rating = testR[2]
        sumRMSE += pow(rating - np.dot(P[userID], Q[itemID]), 2)
        sumCount += 1
    RMSE = sqrt(sumRMSE / sumCount)
    return RMSE

@jit(nopython=True)
def tMAE(P, Q, testSet):
    sumMAE = 0
    sumCount = 0
    for testR in testSet:
        userID = int(testR[0])
        itemID = int(testR[1])
        rating = testR[2]
        sumMAE += abs(rating - np.dot(P[userID], Q[itemID]))
        sumCount += 1
    MAE = sumMAE /sumCount
    return MAE

@jit(nopython=True)
def train(trainSet, testSet, userMaxID, itemMaxID, lam, eta, dim, P, Q, userRSetSize, itemRSetSize):
    print("lambda:", lam)
    print("eta:", eta)
    print(">>>>>>>>>>>>>>>>>>>>>")

    lastErr = 0
    minGap = 1e-5
    minErr = 100
    cachedMinTime = 0
    maxRound = 1000
    minRound = 0
    totalRound = 0
    minTotalTime = 0.0
    totalTime = 0.0

    X = P
    Y = Q
    X_U = np.zeros(userMaxID)
    X_D = np.zeros(userMaxID)
    X_C = np.zeros(userMaxID)
    Y_U = np.zeros(itemMaxID)
    Y_D = np.zeros(itemMaxID)
    Y_C = np.zeros(itemMaxID)
    # Gamma_U = np.zeros((userMaxID, dim))
    # Gamma_I = np.zeros((itemMaxID, dim))
    np.random.seed(0)
    Gamma_U = np.random.uniform(0, 1e-10, (userMaxID, dim))  # 必须是1e-10
    np.random.seed(1)
    Gamma_I = np.random.uniform(0, 1e-10, (itemMaxID, dim))  # 必须是1e-10
    ratingHat = np.zeros((userMaxID, itemMaxID))

    train_size = trainSet.shape[0]

    for trainR in trainSet:
        userID = int(trainR[0])
        itemID = int(trainR[1])
        value = np.dot(X[userID], Y[itemID])
        ratingHat[userID][itemID] = value

    for round in range(maxRound):
        with objmode(startTime='f8'):
            startTime = perf_counter()

        for k in range(dim):
            for id in range(userMaxID):
                X_U[id] = 0.0
                X_D[id] = 0.0
            for id in range(itemMaxID):
                Y_U[id] = 0.0
                Y_D[id] = 0.0

            for i in range(train_size):
                trainR = trainSet[i]  # 取第i行，即取第i个评分
                userID = int(trainR[0])
                itemID = int(trainR[1])
                rating = trainR[2]
                X_U[userID] = X_U[userID] + Y[itemID][k] * (rating - ratingHat[userID][itemID])
                X_U[userID] = X_U[userID] + X[userID][k] * Y[itemID][k] * Y[itemID][k]
                X_D[userID] = X_D[userID] + Y[itemID][k] * Y[itemID][k]

                Y_U[itemID] = Y_U[itemID] + X[userID][k] * (rating - ratingHat[userID][itemID])
                Y_U[itemID] = Y_U[itemID] + Y[itemID][k] * X[userID][k] * X[userID][k]
                Y_D[itemID] = Y_D[itemID] + X[userID][k] * X[userID][k]

            for id in range(userMaxID):  # prange与range的结果一样，同时时间效率也没有提高
                X_C[id] = X[id][k]
                rho = lam * userRSetSize[id]
                X_U[id] = X_U[id] + rho * P[id][k]
                X_U[id] = X_U[id] - Gamma_U[id][k]
                if rho == 0:
                    rho = 1e-8  # 必须是1e-8
                X_D[id] = X_D[id] + rho
                X[id][k] = X_U[id] / X_D[id]

                tempP = X[id][k] + Gamma_U[id][k] / rho
                if tempP > 0:
                    P[id][k] = tempP
                else:
                    P[id][k] = 0

                Gamma_U[id][k] = Gamma_U[id][k] + eta * rho * (X[id][k] - P[id][k])

            for id in range(itemMaxID):  # prange与range的结果一样，同时时间效率也没有提高
                Y_C[id] = Y[id][k]
                tau = lam * itemRSetSize[id]
                Y_U[id] = Y_U[id] + tau * Q[id][k]
                Y_U[id] = Y_U[id] - Gamma_I[id][k]
                if tau == 0:
                    tau = 1e-8  # 必须是1e-8
                Y_D[id] = Y_D[id] + tau
                Y[id][k] = Y_U[id] / Y_D[id]

                tempQ = Y[id][k] + Gamma_I[id][k] / tau
                if tempQ > 0:
                    Q[id][k] = tempQ
                else:
                    Q[id][k] = 0

                Gamma_I[id][k] = Gamma_I[id][k] + eta * tau * (Y[id][k] - Q[id][k])

            for i in range(train_size):
                trainR = trainSet[i]
                userID = int(trainR[0])
                itemID = int(trainR[1])
                value = X[userID][k] * Y[itemID][k] - X_C[userID] * Y_C[itemID]
                ratingHat[userID][itemID] = ratingHat[userID][itemID] + value

        with objmode(endTime='f8'):
            endTime = perf_counter()

        cachedTime = endTime - startTime
        cachedMinTime = cachedMinTime + cachedTime
        totalTime = totalTime + cachedTime

        if metric == "RMSE":
            curErr = tRMSE(P, Q, testSet)  # 所调用函数也必须要成功使用numba加速
        else:
            curErr = tMAE(P, Q, testSet)
        print((round + 1), ":", curErr, ":", cachedTime)

        totalRound = totalRound + 1

        if minErr > curErr:
            minErr = curErr
            minRound = round
            minTotalTime = minTotalTime + cachedMinTime
            cachedMinTime = 0
        else:
            if round - minRound >= delayCount:
                break

        if abs(curErr - lastErr) > minGap:
            lastErr = curErr
        else:
            break

    print("Min Error:\t\t\t", minErr)
    print("Min total training epochs:\t\t\t", minRound + 1)
    print("Total training epochs:\t\t\t", totalRound)
    print("Min total training time:\t\t\t", minTotalTime)
    print("Min average training time:\t\t\t", minTotalTime / (minRound + 1))
    print("Total training time:\t\t\t", totalTime)
    print("Average training time:\t\t\t", totalTime / totalRound)
    print("=======================\n")

    return P, Q, minErr


# hyperopt自适应调參的公共部分
trainSet, valSet, userMaxID, itemMaxID = dataLoad(trainFile, valFile, "::")
userRSetSize, itemRSetSize = initRatingSetSize(trainSet, userMaxID, itemMaxID)
trainSet = np.array(trainSet)
valSet = np.array(valSet)

def hyperopt_Train(lam, eta):

    P, Q = initFeature_Random(userMaxID, itemMaxID, dim)  # 若把FM放在外面变为全局变量的话，这就接近于在自适应调參过程中固定了初始值
    print("=======验证集上的结果==========")
    updated_P, updated_Q, minErr = train(trainSet, valSet, userMaxID, itemMaxID, lam, eta, dim, P, Q, userRSetSize, itemRSetSize)
    return minErr

def hyperopt_objective(params):
    # 定义评估器
    # 需要搜索的参数需要从输入的字典中索引出来
    # 不需要搜索的参数，可以是设置好的某个值
    # 在需要整数的参数前调整参数类型
    reg = hyperopt_Train(lam=params["lam"],
                         eta=params["eta"])
    return reg

param_grid_simple = {
        'lam': hp.uniform("lam", 0.2, 2),
        'eta': hp.uniform("eta", 1, 2)
}

def param_hyperopt(max_evals=100):
    # 保存迭代过程
    trials = Trials()
    # 设置提前停止, 当n轮内loss基本不再变化
    early_stop_fn = no_progress_loss(early_stop)
    # 定义代理模型
    # algo = partial(tpe.suggest, n_startup_jobs=20, n_EI_candidates=50)
    params_best = fmin(hyperopt_objective # 目标函数
                       , space=param_grid_simple # 参数空间
                       , algo=tpe.suggest # 代理模型你要哪个呢？
                       # , algo = algo
                       , max_evals=max_evals # 允许的最大迭代次数
                       , verbose=True
                       , trials=trials
                       , early_stop_fn=early_stop_fn
                       )

    # 打印最优参数，fmin会自动打印最佳分数
    print("\n", "\n", "best params: ", params_best,
          "\n")
    return params_best, trials

if __name__ == '__main__':

    start_time = time.time()
    params_best, trials = param_hyperopt(max_evals)
    end_time = time.time()
    print("Val_Time_Cost=", end_time - start_time)

    trainSet, testSet, userMaxID, itemMaxID = dataLoad(trainFile, testFile, "::")
    userRSetSize, itemRSetSize = initRatingSetSize(trainSet, userMaxID, itemMaxID)
    trainSet = np.array(trainSet)
    testSet = np.array(testSet)

    P, Q = initFeature_Random(userMaxID, itemMaxID, dim)

    print("=======测试集上的结果==========")
    final_P, final_Q, minErr = train(trainSet, testSet, userMaxID, itemMaxID, params_best['lam'], params_best['eta'],
          dim, P, Q, userRSetSize, itemRSetSize)

    Y_fileName = datasetName + "_" + metric + "_dim=" + str(dim) + "_Y.txt"
    YHat_fileName = datasetName + "_" + metric + "_dim=" + str(dim) + "_YHat.txt"
    Y_fw = open(Y_fileName, 'w')
    YHat_fw = open(YHat_fileName, 'w')
    for testR in testSet:
        userID = int(testR[0])
        itemID = int(testR[1])
        rating = testR[2]
        ratingHat = np.dot(final_P[userID], final_Q[itemID])
        Y_fw.write(str(rating) + "\n")
        YHat_fw.write(str(ratingHat) + "\n")
        Y_fw.flush()
        YHat_fw.flush()
    Y_fw.close()
    YHat_fw.close()
