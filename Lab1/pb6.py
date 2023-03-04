"""
Pentru un șir cu n numere întregi care conține și duplicate, să se determine
elementul majoritar (care apare de mai mult de n / 2 ori). De ex. 2 este
elementul majoritar în șirul [2,8,7,2,2,5,2,3,1,2,2].
"""


def majority_elem1(nums: list[int]) -> int:
    '''
    O(n) timp
    O(n) spatiu
    '''
    hash_map = {}
    for num in nums:
        hash_map[num] = hash_map.get(num, 0) + 1
    for num in nums:
        if hash_map[num] > len(nums) // 2:
            return num
    return -1



def majority_elem2(nums: list[int]) -> int:
    '''
    O(n) timp
    O(1) spatiu
    '''
    flag = 0
    maj = -1
    for num in nums:
        if flag == 0:
            maj = num
        if num == maj:
            flag += 1
        else:
            flag -= 1
    return maj



def test(func):
    assert func([7, 7, 5, 7, 5, 1, 5, 7, 5, 5, 7, 7, 7, 7, 7, 7]) == 7
    assert func([7, 7, 5, 7, 5, 1, 5, 7, 5, 5, 7, 7, 5, 5, 5, 5]) == 5
    assert func([2, 8, 7, 2, 2, 5, 2, 3, 1, 2, 2]) == 2
    assert func([3, 2, 3]) == 3
    assert func([2, 2, 1, 1, 1, 2, 2]) == 2
    assert func([1]) == 1


if __name__ == '__main__':
    test(majority_elem1)
    test(majority_elem2)
    rez = majority_elem2([2, 8, 7, 2, 2, 5, 2, 3, 1, 2, 2])
    print(rez)