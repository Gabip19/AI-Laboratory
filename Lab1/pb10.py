"""
Considerându-se o matrice cu n x m elemente binare (0 sau 1) sortate crescător
pe linii, să se identifice indexul liniei care conține cele mai multe elemente de 1.
"""


def row_with_1s1(matrix: list[list[int]]) -> int:
    '''
    Parcurgere pe orizonatala de la stanga la dreapta
    O(n*m) timp
    O(1) spatiu
    '''
    for j in range(len(matrix[0])):
        for i in range(len(matrix)):
            if matrix[i][j] == 1:
                return i
    return -1



def row_with_1s2(matrix: list[list[int]]) -> int:
    '''
    O(n*logm) timp
    O(1) spatiu
    '''
    left = 0
    right = len(matrix[0]) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        counter = 0
        max_row = 0
        
        for i in range(len(matrix)):
            if (matrix[i][mid] == 1):
                counter += 1
                max_row = i
        
        if counter == 1:
            return max_row
        elif counter > 1:
            right = mid - 1
        else:
            left = mid + 1
    
    return -1



def test(func):
    assert func([[0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                 [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                 [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1, 1, 1, 1, 1]]) == 3
    
    assert func([[0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                 [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]) == 1

    assert func([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]) == 5
    
    assert func([[0, 0, 0, 1, 1],
                 [0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1]]) == 1

    assert func([[0, 0, 0, 1, 1],
                 [0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1],
                 [0, 0, 1, 1, 1],
                 [0, 0, 1, 1, 1],
                 [0, 0, 1, 1, 1],
                 [1, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1]]) == 6


if __name__ == '__main__':
    test(row_with_1s1)
    test(row_with_1s2)
    rez = row_with_1s2([[0, 0, 0, 1, 1],
                        [0, 1, 1, 1, 1],
                        [0, 0, 1, 1, 1]])
    print(rez)