"""
Considerându-se o matrice cu n x m elemente întregi și o listă
cu perechi formate din coordonatele a 2 căsuțe din
matrice ((p,q) și (r,s)), să se calculeze suma elementelor din
sub-matricile identificate de fieare pereche.
"""


def submatrix_sums1(matrix: list[list[int]], pairs) -> list[int]:
    '''
    O(n*m*k) timp, unde k e numarul de perechi
    O(1) spatiu
    '''
    sums = []
    for pair in pairs:
        row_1 = pair[0][0]
        col_1 = pair[0][1]
        row_2 = pair[1][0]
        col_2 = pair[1][1]
        current_sum = 0
        for i in range(row_1, row_2 + 1):
            for j in range(col_1, col_2 + 1):
                current_sum += matrix[i][j]
        sums.append(current_sum)
    return sums




def build_sum_matrix(matrix: list[list[int]]) -> list[list[int]]:
    sum_matrix = [[0] * (len(matrix[0]) + 1) for _ in range(len(matrix) + 1)]
    
    for row in range(len(sum_matrix) - 1):
        for col in range(len(sum_matrix[0]) - 1):
            sum_matrix[row + 1][col + 1] = matrix[row][col] + sum_matrix[row][col + 1] + sum_matrix[row + 1][col] - sum_matrix[row][col]
    
    return sum_matrix

def submatrix_sums2(matrix: list[list[int]], pairs) -> list[int]:
    '''
    O(n*m) pt constructia matricei sumelor partiale => O(n*m) timp
    O(n*m) spatiu 
    '''
    sum_matrix = build_sum_matrix(matrix)
    sums = []
    
    for pair in pairs:
        row_1 = pair[0][0]
        col_1 = pair[0][1]
        row_2 = pair[1][0]
        col_2 = pair[1][1]
        current_sum = sum_matrix[row_2 + 1][col_2 + 1] - sum_matrix[row_1][col_2 + 1] - sum_matrix[row_2 + 1][col_1] + sum_matrix[row_1][col_1]
        sums.append(current_sum)
    
    return sums



def test(func):
    assert func([[0, 2, 5, 4, 1],
                 [4, 8, 2, 3, 7],
                 [6, 3, 4, 6, 2],
                 [7, 3, 1, 8, 3],
                 [1, 5, 7, 9, 4]],
                [((1, 1), (3, 3))]) == [38]
    
    assert func([[0, 2, 5, 4, 1],
                 [4, 8, 2, 3, 7],
                 [6, 3, 4, 6, 2],
                 [7, 3, 1, 8, 3],
                 [1, 5, 7, 9, 4]],
                [((2, 2), (4, 4)), ((0, 0), (2, 2))]) == [44, 34]

    assert func([[0, 2, 5, 4, 1],
                 [4, 8, 2, 3, 7],
                 [6, 3, 4, 6, 2],
                 [7, 3, 1, 8, 3],
                 [1, 5, 7, 9, 4]],
                [((0, 0), (0, 0))]) == [0]
    
    assert func([[1, 2, 3, 4],
                 [1, 2, 3, 4],
                 [1, 2, 3, 4]],
                [((0, 0), (1, 2)), ((1, 2), (1, 2))]) == [12, 3]
    
    assert func([[1, 2, 3, 4],
                 [1, 2, 3, 4],
                 [1, 2, 3, 4]],
                [((0, 0), (2, 3)), ((1, 1), (2, 2))]) == [30, 10]


if __name__ == '__main__':
    test(submatrix_sums1)
    test(submatrix_sums2)
    rez = submatrix_sums2([[0, 2, 5, 4, 1],
                          [4, 8, 2, 3, 7],
                          [6, 3, 4, 6, 2],
                          [7, 3, 1, 8, 3],
                          [1, 5, 7, 9, 4]],
                         [((1, 1), (3, 3)), ((2, 2), (4, 4))])
    print(rez)