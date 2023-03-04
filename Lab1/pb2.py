"""
Să se determine distanța Euclideană între două locații identificate prin perechi de numere.
De ex. distanța între (1,5) și (4,1) este 5.0
"""

import math


def distanta_eucld(point1: tuple[int, int], point2: tuple[int, int]) -> float:
    first_square = (point2[0] - point1[0]) ** 2
    second_square = (point2[1] - point1[1]) ** 2
    return math.sqrt(first_square + second_square)


def test(func):
    assert func((1, 5), (4, 1)) == 5.0
    assert func((4, 6), (1, 2)) == 5.0
    assert func((65, 1), (26, 6)) == math.dist((65, 1), (26, 6))
    assert func((6, 0), (-4, 3)) == math.dist((6, 0), (-4, 3))
    assert func((-1, 0), (-1, 0)) == 0


if __name__ == '__main__':
    test(distanta_eucld)
    rez = distanta_eucld((1, 5), (4, 1))
    print(rez)