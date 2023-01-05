import numpy
import Functions.functions
from Functions import functions, GaussianClassifier as GC, LogisticRegression as LR, SVM, GMM

### Fabio Grillo s287873  HTRU2###


# import and load dataset
def load_data(pre = "./Functions/Data/"):
    #DTR, LTR -> training DTE, LTE -> test
    (DTR, LTR), (DTE, LTE) = functions.load_data(pre + "Train.txt", pre + "Test.txt", 8)
    return (DTR, LTR), (DTE, LTE)


# plot features and show distributions
def plot_features(D, L):
    #histograms
    functions.plot_features(D, L)
    #heatmaps
    functions.plot_pCorr(D, L)


# Gaussian classifier
def gaussian_classifier_wrap(DTR, LTR):
    #first 5 folds then single split
    model = GC.GaussianClassifier()
    DTRpca = DTR
    print('# 5-folds')
    for i in range(4):  # raw, pca7, pca6, pca5
        print(f'PCA m = {DTR.shape[0] - i}' if i > 0 else '# RAW')
        if(i > 0):
            PCA_ = functions.PCA(DTR, DTR.shape[0] - i)
            DTRpca = PCA_[0]
        print('Full Covariance')
        for pi in priors:
            minDCF = functions.kfolds(DTRpca, LTR, pi, model, ([pi, 1 - pi]))[0]
            print(f'%.3f' % minDCF)
        print('Diag Covariance')
        for pi in priors:
            minDCF = functions.kfolds(DTRpca, LTR, pi, model, ([pi, 1 - pi], 'NBG'))[0]
            print(f'%.3f' % minDCF)
        print('Tied Full Covariance')
        for pi in priors:
            minDCF = functions.kfolds(DTRpca, LTR, pi, model, ([pi, 1 - pi], 'MVG', True))[0]
            print(f'%.3f' % minDCF)
        print('Tied Diagonal Covariance')
        for pi in priors:
            minDCF = functions.kfolds(DTRpca, LTR, pi, model, ([pi, 1 - pi], 'NBG', True))[0]
            print(f'%.3f' % minDCF)
    print('\n')

    print('# single-split')
    for i in range(4):  # raw, pca7, pca6, pca5
        print(f'PCA m = {DTR.shape[0] - i}' if i > 0 else '# RAW')
        if(i > 0):
            PCA_ = functions.PCA(DTR, DTR.shape[0] - i)
            DTRpca = PCA_[0]
        print('Full Covariance')
        for pi in priors:
            minDCF = functions.single_split(DTRpca, LTR, pi, model, ([pi, 1 - pi]))[0]
            print(f'%.3f' % minDCF)
        print('Diagonal Covariance')
        for pi in priors:
            minDCF = functions.single_split(DTRpca, LTR, pi, model, ([pi, 1 - pi], 'NBG'))[0]
            print(f'%.3f' % minDCF)
        print('Tied Full Covariance')
        for pi in priors:
            minDCF = functions.single_split(DTRpca, LTR, pi, model,  ([pi, 1 - pi], 'MVG', True))[0]
            print(f'%.3f' % minDCF)
        print('Tied Diagonal Covariance')
        for pi in priors:
            minDCF = functions.single_split(DTRpca, LTR, pi, model, ([pi, 1 - pi], 'NBG', True))[0]
            print(f'%.3f' % minDCF)


# Logistic Regression
def logistic_regression_wrap(D, L):
    model = LR.LogisticRegression()
    DTR_pca = D

    l = numpy.logspace(-5, 1, 10)
    for i in range(3): #raw, pca7, pca6
        y5, y1, y9= [], [], []
        #title = "raw"
        if(i > 0):
            PCA_ = functions.PCA(D, D.shape[0] - i)
            DTR_pca = PCA_[0]
            #title = f'pca{D.shape[0] - i}'
        for item in l:
            y5.append(functions.kfolds(DTR_pca, L, priors[0], model, (item, priors[0]))[0])
            y1.append(functions.kfolds(DTR_pca, L, priors[1], model, (item, priors[0]))[0])
            y9.append(functions.kfolds(DTR_pca, L, priors[2], model, (item, priors[0]))[0])
        #functions.plot_minDCF_lr(l, y5, y1, y9, f'{title}_5-folds', f'5-folds / {title} / πT = 0.5')

    for i in range(3):
        print(f'PCA m = {DTR.shape[0] - i}' if i > 0 else '# RAW')
        if (i > 0):PCA_ = functions.PCA(D, D.shape[0] - i)
        DTR_pca = PCA_[0]
        print('lambda = 1e-5, piT = 0.5)')
        for pi in priors:
            minDCF = functions.kfolds(DTR_pca, L, pi, model, (1e-5, priors[0]))[0]
            print(f'%.3f' % minDCF)
        print('lambda = 1e-5, piT = 0.1)')
        for pi in priors:
            minDCF = functions.kfolds(DTR_pca, L, pi, model, (1e-5, priors[1]))[0]
            print(f'%.3f' % minDCF)
        print('lambda = 1e-5, piT = 0.9)')
        for pi in priors:
            minDCF = functions.kfolds(DTR_pca, L, pi, model, (1e-5, priors[2]))[0]
            print(f'%.3f' % minDCF)


# Support Vector Machine - linear
def linear_svm_wrap(DTR, LTR):
    model = SVM.SVM()
    DTRpca = DTR

    C = numpy.logspace(-4, 1, 10)
    for mode in ['unbalanced', 'balanced']:
        for i in priors:
            y5, y1, y9 = [], [], []
            PCA_ = functions.PCA(DTR, 7)
            DTRpca = PCA_[0]
            #title = f'pca7'
            for iC in C:
                y5.append(functions.kfolds(DTRpca, LTR, priors[0], model, ('linear', i, mode == 'balanced', 1, iC))[0])
                y1.append(functions.kfolds(DTRpca, LTR, priors[1], model, ('linear', i, mode == 'balanced', 1, iC))[0])
                y9.append(functions.kfolds(DTRpca, LTR, priors[2], model, ('linear', i, mode == 'balanced', 1, iC))[0])
            #functions.plot_minDCF_svm(C, y5, y1, y9, f'linear_{title}_{mode}{i}_5-folds', f'5-folds / {title} / {f"πT = {i}" if mode == "balanced" else "unbalanced"}')
            if(mode == 'unbalanced'):
                break

    for i in range(2):  # raw, pca7
        print(f'# PCA m = {DTR.shape[0] - i}' if i > 0 else '# RAW')
        if(i > 0):
            PCA_ = functions.PCA(DTR, DTR.shape[0] - i)
            DTRpca = PCA_[0]
        print('C = 1e-2')
        for pi in priors:
            minDCF = functions.kfolds(DTRpca, LTR, pi, model, ('linear', priors[0], False, 1, 1e-2))[0]
            print(f'%.3f' % minDCF)
        print('C = 1e-2, piT = 0.5')
        for pi in priors:
            minDCF = functions.kfolds(DTRpca, LTR, pi, model, ('linear', priors[0], True, 1, 1e-2))[0]
            print(f'%.3f' % minDCF)
        print('C = 1e-2, piT = 0.1')
        for pi in priors:
            minDCF = functions.kfolds(DTRpca, LTR, pi, model, ('linear', priors[1], True, 1, 1e-2))[0]
            print(f'%.3f' % minDCF)
        print('C = 1e-2, piT = 0.9')
        for pi in priors:
            minDCF = functions.kfolds(DTRpca, LTR, pi, model, ('linear', priors[2], True, 1, 1e-2))[0]
            print(f'%.3f' % minDCF)


# RBF SVM, Poly SVM
def quadratic_svm_wrap(DTR, LTR):

    model = SVM.SVM()
    DTRpca = DTR

    C = numpy.logspace(-4, 1, 10)
    for i in range(2):  # raw, pca7
        y5, y1, y9 = [], [], []
        #title = 'raw'
        if(i > 0):
            PCA_ = functions.PCA(DTR, DTR.shape[0] - i)
            DTRpca = PCA_[0]
            #title = f'pca{DTR.shape[0] - i}'
        for iC in C:
            y5.append(functions.kfolds(DTRpca, LTR, priors[0], model, ('poly', priors[0], False, 1, iC, 1, 2))[0])
            y1.append(functions.kfolds(DTRpca, LTR, priors[1], model, ('poly', priors[0], False, 1, iC, 10, 2))[0])
            y9.append(functions.kfolds(DTRpca, LTR, priors[2], model, ('poly', priors[0], False, 1, iC, 100, 2))[0])
        #functions.plot_minDCF_svm(C, y5, y1, y9, f'poly_{title}_unbalanced_5-folds', f'5-folds / {title} / unbalanced', type='poly')
    for i in range(2):  # raw, pca7
        y5, y1, y9 = [], [], []
        #title = 'raw'
        if(i > 0):
            PCA_ = functions.PCA(DTR, DTR.shape[0] - i)
            DTRpca = PCA_[0]
            #title = f'pca{DTR.shape[0] - i}'
        for iC in C:
            y5.append(functions.kfolds(DTRpca, LTR, priors[0], model, ('RBF', priors[0], False, 1, iC, 0, 0, 1e-3))[0])
            y1.append(functions.kfolds(DTRpca, LTR, priors[1], model, ('RBF', priors[0], False, 1, iC, 0, 0, 1e-2))[0])
            y9.append(functions.kfolds(DTRpca, LTR, priors[2], model, ('RBF', priors[0], False, 1, iC, 0, 0, 1e-1))[0])
        #functions.plot_minDCF_svm(C, y5, y1, y9, f'rbf_{title}_unbalanced_5-folds', f'5-folds / {title} / unbalanced', type='RBF')

    for i in range(2):  # raw, pca7
        print(f'# PCA m = {DTR.shape[0] - i}' if i > 0 else '# RAW')
        if(i > 0):
            PCA_ = functions.PCA(DTR, DTR.shape[0] - i)
            DTRpca = PCA_[0]
        print('RBF C = 1e-1, gamma = 1e-3)')
        for pi in priors:
            minDCF = functions.kfolds(DTRpca, LTR, pi, model, ('RBF', priors[0], False, 1, 1e-1, 0, 0, 1e-3))[0]
            print(f'%.3f' % minDCF)
        print('Poly C = 1e-3, c = 1, d = 2)')
        for pi in priors:
            minDCF = functions.kfolds(DTRpca, LTR, pi, model, ('poly', priors[0], False, 1, 1e-3, 1, 2, 0))[0]
            print(f'%.3f' % minDCF)


# GMM
def gmm_wrap(DTR, LTR):

    model = GMM.GMM()
    DTR_pca = DTR

    components = [2,4,8,16,32]
    for type in ["full", "tied", "diag"]:
        for i in range(2):
            y5, y1, y9 = [], [], []
            #title = "raw"
            if(i > 0):
                PCA_ = functions.PCA(DTR, DTR.shape[0] - i)
                DTR_pca = PCA_[0]
                #title = f'pca{DTR.shape[0] - i}'

            for c in components:
                y5.append(functions.kfolds(DTR_pca, LTR, priors[0], model, (c, type))[0])
                y1.append(functions.kfolds(DTR_pca, LTR, priors[1], model, (c, type))[0])
                y9.append(functions.kfolds(DTR_pca, LTR, priors[2], model, (c, type))[0])
            #functions.plot_minDCF_gmm(components, y5, y1, y9, f'{type}_{title}', f'gmm {type}-cov / {title}')


    for i in range(2):
        print(f'# PCA m = {DTR.shape[0] - i}' if i > 0 else '# RAW')
        if (i > 0):
            PCA_ = functions.PCA(DTR, DTR.shape[0] - i)
            DTR_pca = PCA_[0]
        print('GMM Full (8 components)')
        for pi in priors:
            minDCF = functions.kfolds(DTR_pca, LTR, pi, model, (8, 'full'))[0]
            print(f'%.3f' % minDCF)
        print('GMM Diag (16 components)')
        for pi in priors:
            minDCF = functions.kfolds(DTR_pca, LTR, pi, model, (16, 'diag'))[0]
            print(f'%.3f' % minDCF)
        print('GMM Tied (32 components)')
        for pi in priors:
            minDCF = functions.kfolds(DTR_pca, LTR, pi, model, (32, 'tied'))[0]
            print(f'%.3f' % minDCF)


# Score calibration
def score_calibration_wrap(DTR, LTR):
    DTRpca = DTR
    p = numpy.linspace(-3, 3, 15)
    for model in [
            (GC.GaussianClassifier(), ([priors[0], 1 - priors[0]], 'MVG', True), 'tiedFullCov', 'Tied Full-Cov / PCA = 7'),
            (LR.LogisticRegression(), (1e-5, priors[0]), 'LogReg', 'Logistic Regression / λ = 1e-5 / PCA = 7'),
            (SVM.SVM(), ('linear', priors[0], False, 1, 1e-2), 'SVM','Linear SVM / C = 1e-2 / PCA = 7'),
            (GMM.GMM(), (8, 'full'), 'GMM','Tied GMM / 8 components / PCA = 7'),
        ]:
        minDCF = []
        actDCF = []
        for iP in p:
            iP = 1.0 / (1.0 + numpy.exp(-iP))
            minDCFtmp, actDCFtmp = functions.kfolds(DTRpca, LTR, iP, model[0], model[1])
            minDCF.append(minDCFtmp)
            actDCF.append(actDCFtmp)
        #functions.bayes_error_plot(p, minDCF, actDCF, model[2], model[3])

    p = numpy.linspace(-3, 3, 15)
    for model in [
            (GC.GaussianClassifier(), ([priors[0], 1 - priors[0]], 'MVG', True), 'calibrated_tiedFullCov', 'calibrated Tied Full-Cov / PCA = 7'),
            (LR.LogisticRegression(), (1e-5, priors[0]), 'calibrated_LogReg', 'calibrated  Logistic Regression / λ = 1e-5 / PCA = 7'),
            (SVM.SVM(), ('linear', priors[0], False, 1, 1e-2), 'calibrated_SVM','calibrated  Linear SVM / C = 1e-2 / PCA = 7'),
            (GMM.GMM(), (8, 'full'), 'calibrated_GMM','calibrated  Tied GMM / 8 components / PCA = 7'),
        ]:
        minDCF = []
        actDCF = []
        for iP in p:
            iP = 1.0 / (1.0 + numpy.exp(-iP))
            minDCFtmp, actDCFtmp = functions.kfolds(DTRpca, LTR, iP, model[0], model[1], calibrated=True)
            minDCF.append(minDCFtmp)
            actDCF.append(actDCFtmp)
        functions.bayes_error_plot(p, minDCF, actDCF, model[2], model[3])
    print('#5-folds')
    for i in range(2):  # raw, pca7
        print(f'# PCA m = {DTR.shape[0] - i}' if i > 0 else '# RAW')
        if(i > 0):
            PCA_ = functions.PCA(DTR, DTR.shape[0] - i)
            DTRpca = PCA_[0]
        print('Tied Full Covariance')
        for pi in priors:
            minDCF, actDCF = functions.kfolds(DTRpca, LTR, pi, GC.GaussianClassifier(), ([priors[0], 1 - priors[0]], 'MVG', True), calibrated=True)
            print(f'min = %.3f' % minDCF)
            print(f'act = %.3f' % actDCF)
        print('lambda = 1e-5, piT = 0.5)')
        for pi in priors:
            minDCF, actDCF = functions.kfolds(DTRpca, LTR, pi, LR.LogisticRegression(), (1e-5, priors[0]), calibrated=True)
            print(f'min = %.3f' % minDCF)
            print(f'act = %.3f' % actDCF)
        print('SVM C = 1e-2, piT = 0.5)')
        for pi in priors:
            minDCF, actDCF = functions.kfolds(DTRpca, LTR, pi, SVM.SVM(), ('linear', priors[0], False, 1, 1e-2), calibrated=True)
            print(f'min = %.3f' % minDCF)
            print(f'act = %.3f' % actDCF)
        print('GMM Full Covariance - 8 components')
        for pi in priors:
            minDCF, actDCF = functions.kfolds(DTRpca, LTR, pi, GMM.GMM(), (8, 'full'), calibrated=True)
            print(f'min = %.3f' % minDCF)
            print(f'act = %.3f' % actDCF)

# Evaluation
def evaluation_wrap(DTR, LTR, DTE, LTE):
    PCA_ = functions.PCA(DTR, 7)
    DTRpca = PCA_[0]
    DTEpca = numpy.dot(PCA_[1].T, DTE)
    calibratedScores = []
    for model in [
            (GC.GaussianClassifier().trainClassifier(DTRpca, LTR, *([priors[0], 1 - priors[0]], 'MVG', True)), 'Tied Full Covariance'),
            (LR.LogisticRegression().trainClassifier(DTRpca, LTR, *(1e-5, priors[0])), 'LR lambda = 1e-5, piT = 0.5'),
            (SVM.SVM().trainClassifier(DTRpca, LTR, *('linear', priors[0], False, 1, 1e-2)), 'SVM C = 1e-2'),
            (GMM.GMM().trainClassifier(DTRpca, LTR, *(8, 'full')), 'GMM Full Cov - 8 components')
        ]:
        alpha, beta = functions.compute_calibrated_scores_param(model[0].computeLLR(DTRpca), LTR)
        scores = alpha * model[0].computeLLR(DTEpca) + beta - numpy.log(priors[0]/(1 - priors[0]))
        print(model[1])
        for pi in priors:
            minDCF = functions.minDCF(scores, LTE, pi, 1, 1)
            actDCF = functions.actDCF(scores, LTE, pi, 1, 1)
            print(f'min = %.3f' % minDCF)
            print(f'act = %.3f' % actDCF)
        calibratedScores.append(scores)
    #functions.plot_ROC(zip(calibratedScores, ['Tied Full-Cov','LogReg(λ = 1e-5, πT = 0.5)','Linear SVM(C = 1e-2)','GMM Full Cov (8 components)'], ['r','b','g','darkorange']), LTE, 'calibrated_classifiers', 'calibrated / PCA = 7')
    #functions.plot_DET(zip(calibratedScores, ['Tied Full-Cov','LogReg(λ = 1e-5, πT = 0.5)','Linear SVM(C = 1e-2)','GMM Full Cov (8 components)'], ['r','b','g','darkorange']), LTE, 'calibrated_classifiers', 'calibrated / PCA = 7')


if __name__ == '__main__':
    (DTR, LTR), (DTE, LTE) = load_data()
    plot_features(DTR, LTR)
    priors = [0.5, 0.1, 0.9]

    print("Starting Gaussian Classifier\n")
    gaussian_classifier_wrap(DTR, LTR)
    print("GC finished.\n")

    print("Starting Logistic Regression\n")
    logistic_regression_wrap(DTR, LTR)
    print("LR finished.\n")

    print("Starting linear SVM\n")
    linear_svm_wrap(DTR, LTR)
    print("Linear SVM finished.\n")

    print("Starting quadratic SVM (Kernel RBF, Polynomial)\n")
    quadratic_svm_wrap(DTR, LTR)
    print("Quadratic SVM finished.\n")

    print("Starting Gaussian Mixture Model\n")
    gmm_wrap(DTR, LTR)
    print("GMM finished.\n")

    print("Starting Score Calibration\n")
    score_calibration_wrap(DTR, LTR)
    print("Score Calibration finished.\n")

    print("Starting Evaluation\n")
    evaluation_wrap(DTR, LTR, DTE, LTE)
    print("Evaluation finished.\n")
