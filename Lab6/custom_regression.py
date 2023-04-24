
class CustomLiniarBivariateRegression:
    def __init__(self):
        self.w = []

    def fit(self, features, result):  # w = (XT * X) ** (-1) * (XT) * Y
        X = [[1] + feature for feature in features]
        XT = self.transpose_matrix(X)
        XTX = self.multiply(XT, X)
        XTX_inverse = self.invert_matrix(XTX)
        XTX_inverse_XT = self.multiply(XTX_inverse, XT)
        Y = [[value] for value in result]
        XTX_inverse_XTY = self.multiply(XTX_inverse_XT, Y)
        self.w = [value[0] for value in XTX_inverse_XTY]

    @staticmethod
    def multiply(m1, m2):
        mat = [[0 for _2 in range(len(m2[0]))] for _1 in range(len(m1))]
        for i in range(len(m1)):
            for j in range(len(m2[0])):
                for x in range(len(m1[0])):
                    mat[i][j] += m1[i][x] * m2[x][j]
        return mat

    @staticmethod
    def transpose_matrix(matrix):
        return list(map(list, zip(*matrix)))

    def predict(self, features):
        y = []
        for feature in features:
            y.append(self.w[0] + feature[0] * self.w[1] + feature[1] * self.w[2])
        return y

    def invert_matrix(self, matrix):
        mc = self.copy_matrix(matrix)
        im = self.get_identity_matrix(len(matrix), len(matrix[0]))
        imc = self.copy_matrix(im)
        indices = list(range(len(matrix)))
        for fd in range(len(matrix)):
            fdScaler = 1 / mc[fd][fd]
            for j in range(len(matrix)):
                mc[fd][j] *= fdScaler
                imc[fd][j] *= fdScaler
            for i in indices[0:fd] + indices[fd + 1:]:
                crScaler = mc[i][fd]
                for j in range(len(matrix)):
                    mc[i][j] = mc[i][j] - crScaler * mc[fd][j]
                    imc[i][j] = imc[i][j] - crScaler * imc[fd][j]
        return imc

    def get_identity_matrix(self, rows, cols):
        identity = self.init_matrix(rows, cols)
        for i in range(rows):
            identity[i][i] = 1
        return identity

    def copy_matrix(self, matrix):
        mc = self.init_matrix(len(matrix), len(matrix[0]))
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                mc[i][j] = matrix[i][j]
        return mc

    @staticmethod
    def init_matrix(rows, cols):
        matrix = []
        for _ in range(rows):
            matrix += [[0 for _ in range(cols)]]
        return matrix
