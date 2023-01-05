import numpy
import scipy.optimize
import Functions.functions as functions


def features_expansion(D):
    expansion = []
    for i in range(D.shape[1]):
        vec = numpy.reshape(numpy.dot(functions.vcol(D[:, i]),functions.vcol(D[:, i]).T), (-1,1), order="F")
        expansion.append(vec)

    return numpy.vstack((numpy.hstack(expansion), D))


def logreg_obj_wrap(D, L, l, pi):
    Z = (L * 2) - 1
    M = D.shape[0]

    def logreg_obj(v):
        w, b = functions.vcol(v[0:M]), v[-1]
        t1 = 0.5 * l * (numpy.linalg.norm(w) ** 2)
        t2 = ((pi) * (L[L == 1].shape[0] ** -1)) * numpy.logaddexp(0, -Z[Z == 1] * (numpy.dot(w.T, D[:, L == 1]) + b)).sum()
        t3 = ((1 - pi) * (L[L == 0].shape[0] ** -1)) * numpy.logaddexp(0, -Z[Z == -1] * (numpy.dot(w.T, D[:, L == 0]) + b)).sum()
        return t1 + t2 + t3
    return logreg_obj


class LogisticRegression:

    def trainClassifier(self, D, L, l, pi, type="linear"):
        self.LTR = L
        self.type = type
        DT = features_expansion(D) if type == "quadratic" else D
        K = L.max() + 1
        M = DT.shape[0]
        obj = logreg_obj_wrap(DT, L, l, pi)
        x, f, d = scipy.optimize.fmin_l_bfgs_b(
            obj,
            x0=numpy.zeros(M * K + K),
            approx_grad=True
        )
        self.w, self.b = x[0:M], x[-1]
        return self


    def computeLLR(self, D):
        DE = features_expansion(D) if self.type == "quadratic" else D
        postLLR = numpy.dot(self.w, DE) + self.b
        return postLLR - numpy.log(self.LTR[self.LTR == 1].shape[0] / self.LTR[self.LTR == 0].shape[0])