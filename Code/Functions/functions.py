import numpy
import scipy.stats
import matplotlib.pyplot as plt
import Functions.LogisticRegression as LR

# import dataset and load
def load_data(f1, f2, numFeat):

    dataList = []
    labelList = []

    file1 = open(f1, "r")
    file2 = open(f2, "r")

    #read features for each line and create numpy array
    for line in file1:
        feat = line.split(",")[0:numFeat]
        feat = numpy.array([i for i in feat])
        feat = vcol(feat)
        dataList.append(feat)
        # cl -> pulsar or interference class
        cl = line.split(",")[-1].strip()
        labelList.append(cl)
    DTR = numpy.hstack(numpy.array(dataList, dtype=numpy.float32))
    DTRm = empirical_mean(DTR)
    # (x_i - mu) / sigma
    DTRstd = vcol(numpy.std(DTR, axis=1))
    DTR = (DTR - DTRm) / DTRstd
    LTR = numpy.array(labelList, dtype=numpy.int32)
    # print(DTR.shape, LTR.shape)
    file1.close()

    # same for file 2 (test)
    dataList = []
    labelList = []

    for line in file2:
        feat = line.split(",")[0:numFeat]
        feat = numpy.array([i for i in feat])
        feat = vcol(feat)
        dataList.append(feat)
        cl = line.split(",")[-1].strip()
        labelList.append(cl)
    DTE = numpy.hstack(numpy.array(dataList, dtype=numpy.float32))
    DTE = (DTE - DTRm) / DTRstd
    LTE = numpy.array(labelList, dtype=numpy.int32)
    # print(DTE.shape, LTE.shape)
    file2.close()

    return shuffle(DTR, LTR), (DTE, LTE)


def split_db_2to1(D,L):
    nTrain = int(D.shape[1] * 2. / 3.)
    numpy.random.seed(0)
    index = numpy.random.permutation(D.shape[1])
    idxTrain = index[0:nTrain]
    idxTest = index[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)


def shuffle(D, L):
    numpy.random.seed(0)
    idx = numpy.random.permutation(D.shape[1])
    return D[:, idx], L[idx]


def vrow(v):
    return v.reshape(1, v.size)


def vcol(v):
    return v.reshape(v.size, 1)


def empirical_mean(X):
    return vcol(X.mean(1))


def empirical_covariance(X):
    mu = empirical_mean(X)
    C = numpy.dot((X - mu), (X - mu).T) / X.shape[1]
    return C


def empirical_withinclass_cov(D, labels):
    SW = 0
    for i in set(list(labels)):
        X = D[:, labels == i]
        SW += X.shape[1] * empirical_covariance(X)
    return SW / D.shape[1]


def empirical_betweenclass_cov(D, labels):
    SB = 0
    muGlob = empirical_mean(D)  # mean of the dataset
    for i in set(list(labels)):
        X = D[:, labels == i]
        mu = empirical_mean(X)  # mean of the class
        SB += X.shape[1] * numpy.dot((mu - muGlob), (mu - muGlob).T)
    return SB / D.shape[1]


def assign_labels(scores, pi, Cfn, Cfp, th=None):
    if th is None:
        th = - numpy.log(pi * Cfn) + numpy.log((1 - pi) * Cfp)
    P = scores > th
    return numpy.int32(P)


def conf_matrix(Pred, labels):
    C = numpy.zeros((2, 2))
    C[0, 0] = ((Pred == 0) * (labels == 0)).sum()
    C[0, 1] = ((Pred == 0) * (labels == 1)).sum()
    C[1, 0] = ((Pred == 1) * (labels == 0)).sum()
    C[1, 1] = ((Pred == 1) * (labels == 1)).sum()
    return C


def PCA(D, m):
    DC = (D - empirical_mean(D))
    C = (1 / DC.shape[1]) * numpy.dot(DC, DC.T)
    s, U = numpy.linalg.eigh(C)
    P = U[:, ::-1][:, 0:m]
    return numpy.dot(P.T, D), P


def DCF(Conf, pi, Cfn, Cfp):
    _DCFu = DCFu(Conf, pi, Cfn, Cfp)
    return _DCFu / min(pi * Cfn, (1 - pi) * Cfp)


def DCFu(Conf, pi, Cfn, Cfp):
    FNR = Conf[0, 1]/(Conf[0, 1] + Conf[1, 1])
    FPR = Conf[1, 0]/(Conf[0, 0] + Conf[1, 0])
    return pi * Cfn * FNR + (1 - pi) * Cfp * FPR


def minDCF(scores, labels, pi, Cfn, Cfp):
    t = numpy.array(scores)
    t.sort()

    dcfList = []
    for _th in t:
        dcfList.append(actDCF(scores, labels, pi, Cfn, Cfp, th=_th))
    return numpy.array(dcfList).min()


def actDCF(scores, labels, pi, Cfn, Cfp, th=None):
    Pred = assign_labels(scores, pi, Cfn, Cfp, th=th)
    CM = conf_matrix(Pred, labels)
    return DCF(CM, pi, Cfn, Cfp)


def plot_features(D, L):
    D0 = D[:, L==0]
    D1 = D[:, L==1]

    labels = {
        0: 'Mean of the integrated profile',
        1: 'Standard deviation of the integrated profile',
        2: 'Excess kurtosis of the integrated profile',
        3: 'Skewness of the integrated profile',
        4: 'Mean of the DM-SNR curve',
        5: 'Standard deviation of the DM-SNR curve',
        6: 'Excess kurtosis of the DM-SNR curve',
        7: 'Skewness of the DM-SNR curve',
    }
    nLab = D.shape[0]
    for i in range(nLab):
        fig = plt.figure()
        plt.title(labels[i])
        plt.hist(D0[i, :], bins=60, density=True, alpha=0.7, label="Negative signal", color="coral", edgecolor="orangered")
        plt.hist(D1[i, :], bins=60, density=True, alpha=0.7, label="Positive signal", color="yellowgreen", edgecolor="darkolivegreen")
        plt.legend()
        #plt.savefig("./Functions/Images/%d_%s" % (i, labels[i]), dpi=400, bbox_inches="tight")
        plt.close(fig)
        # plt.show()


def plot_pCorr(D, L):
    #Pearson's corr. coeff.
    heat_map = ["Greys", "YlOrRd", "YlGn"]
    labels = {
        0: 'Whole Dataset',
        1: 'Negative Signal',
        2: 'Positive Signal'
    }
    CorrCoeff = {
        0: numpy.abs(numpy.corrcoef(D)),
        1: numpy.abs(numpy.corrcoef(D[:, L==0])),
        2: numpy.abs(numpy.corrcoef(D[:, L==1]))
    }
    for i in range(len(CorrCoeff)):
        fig = plt.figure()
        plt.title(labels[i])
        plt.imshow(CorrCoeff[i], cmap=heat_map[i], interpolation="nearest")
        #plt.savefig("./Functions/Images/heatmap_%d_%s" % (i, labels[i]), dpi=400, bbox_inches="tight")
        # plt.show()
        plt.close(fig)


def compute_rates_values(scores, labs):
    t = numpy.array(scores)
    t.sort()
    t = numpy.concatenate([numpy.array([-numpy.inf]), t, numpy.array([numpy.inf])])
    FPR = []
    FNR = []
    for threshold in t:
        Pred = numpy.int32(scores > threshold)
        Conf = conf_matrix(Pred, labs)
        FPR.append(Conf[1, 0] / (Conf[1, 0] + Conf[0, 0]))
        FNR.append(Conf[0, 1] / (Conf[0, 1] + Conf[1, 1]))
    return numpy.array(FPR), numpy.array(FNR), 1 - numpy.array(FPR), 1 - numpy.array(FNR)


def compute_calibrated_scores_param(scores, labels):
    scores = vrow(scores)
    model = LR.LogisticRegression().trainClassifier(scores, labels, 1e-4, 0.5)
    alpha = model.w
    beta = model.b
    return alpha, beta


def kfolds(D, L, pi, model, args, calibrated=False, folds = 5, Cfn = 1, Cfp = 1):
    scores = []
    Ds = numpy.array_split(D, folds, axis=1)
    Ls = numpy.array_split(L, folds)

    for i in range(folds):
        DTRk, LTRk = numpy.hstack(Ds[:i] + Ds[i+1:]), numpy.hstack(Ls[:i] + Ls[i+1:])
        DTEk, LTEk = numpy.asanyarray(Ds[i]), numpy.asanyarray(Ls[i])
        if calibrated:
            scoresTrain = model.trainClassifier(DTRk, LTRk, *args).computeLLR(DTRk)
            alpha, beta = compute_calibrated_scores_param(scoresTrain, LTRk)
            scoresEval = model.computeLLR(DTEk)
            computeLLR = alpha * scoresEval + beta - numpy.log(0.5/(1 - 0.5))
        else:
            computeLLR = model.trainClassifier(DTRk, LTRk, *args).computeLLR(DTEk)
        scores.append(computeLLR)
    minDCFtmp = minDCF(numpy.hstack(scores), L, pi, Cfn, Cfp)
    actDCFtmp = actDCF(numpy.hstack(scores), L, pi, Cfn, Cfp)
    return minDCFtmp, actDCFtmp


def single_split(D, L, pi, model, args, calibrated=False, Cfn = 1, Cfp = 1):
    (DTRk, LTRk), (DTEk, LTEk) = split_db_2to1(D, L)
    scores = []
    if calibrated:
        scoresTrain = model.trainClassifier(DTRk, LTRk, *args).computeLLR(DTRk)
        alpha, beta = compute_calibrated_scores_param(scoresTrain, LTRk)
        scoresEval = model.computeLLR(DTEk)
        scores = alpha * scoresEval + beta - numpy.log(0.5/(1 - 0.5))
    else:
        scores = model.trainClassifier(DTRk, LTRk, *args).computeLLR(DTEk)
    minDCFtmp = minDCF(numpy.hstack(scores), LTEk, pi, Cfn, Cfp)
    actDCFtmp = actDCF(numpy.hstack(scores), LTEk, pi, Cfn, Cfp)
    return minDCFtmp, actDCFtmp


def plot_minDCF_lr(l, y5, y1, y9, fn, title):
    fig = plt.figure()
    plt.title(title)
    plt.plot(l, numpy.array(y5), label="minDCF(π~ = 0.5)", color='r')
    plt.plot(l, numpy.array(y1), label="minDCF(π~ = 0.1)", color='b')
    plt.plot(l, numpy.array(y9), label="minDCF(π~ = 0.9)", color='g')
    plt.xscale("log")
    plt.ylim([0,1])
    plt.xlabel("λ")
    plt.ylabel("minDCF")
    plt.legend()
    #plt.show()
    #plt.savefig("./Functions/Images/LogReg_%s" % (fn))
    plt.close(fig)


def plot_minDCF_svm(C, y5, y1, y9, fn, title, type='linear'):
    labels = {
        0: 'minDCF(π~ = 0.5)' if type == 'linear' else ('minDCF(π~ = 0.5, γ = 1e-3)' if type == 'RBF' else 'minDCF(π~ = 0.5, c = 1)'),
        1: 'minDCF(π~ = 0.1)' if type == 'linear' else ('minDCF(π~ = 0.5, γ = 1e-2)' if type == 'RBF' else 'minDCF(π~ = 0.5, c = 10)'),
        2: 'minDCF(π~ = 0.9)' if type == 'linear' else ('minDCF(π~ = 0.5, γ = 1e-1)' if type == 'RBF' else 'minDCF(π~ = 0.5, c = 100)'),
    }
    fig = plt.figure()
    plt.title(title)
    plt.plot(C, numpy.array(y5), label=labels[0], color='r')
    plt.plot(C, numpy.array(y1), label=labels[1], color='b')
    plt.plot(C, numpy.array(y9), label=labels[2], color='g')
    plt.xscale('log')
    plt.ylim([0, 1])
    plt.xlabel('C')
    plt.ylabel('minDCF')
    plt.legend()
    #plt.savefig('./Functions/Images/svm_minDCF_%s' % fn)
    #plt.show()
    plt.close(fig)


def plot_minDCF_gmm(components, y5, y1, y9, fn, title):
    fig = plt.figure()
    plt.title(title)
    plt.plot(components, numpy.array(y5), label = "minDCF(π~ = 0.5)", color="r")
    plt.plot(components, numpy.array(y1), label = "minDCF(π~ = 0.1)", color="b")
    plt.plot(components, numpy.array(y9), label = "minDCF(π~ = 0.9)", color="g")
    plt.ylim([0, 1])
    plt.xlabel("components")
    plt.ylabel("minDCF")
    plt.legend()
    #plt.savefig("./Functions/Images/gmm_minDCF_%s" % fn)
    #plt.show()
    plt.close(fig)


def bayes_error_plot(p, minDCF, actDCF, fn, title):
    fig = plt.figure()
    plt.title(title)
    plt.plot(p, numpy.array(actDCF), label="actDCF", color="r")
    plt.plot(p, numpy.array(minDCF), label="minDCF", color="b", linestyle="--")
    plt.ylim(([0,1]))
    plt.xlim([-3, 3])
    plt.xlabel("Prior")
    plt.ylabel("minDCF")
    plt.legend()
    #plt.savefig("./Functions/Images/bep_%s" % fn)
    #plt.show()
    plt.close(fig)


def plot_ROC(results, LTE, fn, title):
    fig = plt.figure()
    plt.title(title)
    for result in results:
        FPR, FNR, TNR, TPR = compute_rates_values(result[0], LTE)
        plt.plot(FPR, TPR, label = result[1], color=result[2])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    #plt.savefig('./Functions/Images/roc_%s' % fn)
    #plt.show()
    plt.close(fig)


def plot_DET(results, LTE, fn, title):
    fig = plt.figure()
    plt.title(title)
    for result in results:
        FPR, FNR, TNR, TPR = compute_rates_values(result[0], LTE)
        plt.plot(FPR, FNR, label = result[1], color=result[2])
    plt.xlabel('False Positive Rate')
    plt.ylabel('False Negative Rate')
    plt.legend()
    #plt.savefig('./Functions/Images/det_%s' % fn)
    #plt.show()
    plt.close(fig)