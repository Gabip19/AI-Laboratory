"""
Pentru un șir cu n elemente care conține valori din mulțimea {1, 2, ..., n - 1}
astfel încât o singură valoare se repetă de două ori, să se identifice acea valoare
care se repetă. De ex. în șirul [1,2,3,4,2] valoarea 2 apare de două ori.
"""


def find_duplicate1(nums: list[int]) -> int:
    '''
    O(n) timp
    O(n) spatiu
    '''
    hash_map = {}
    for num in nums:
        hash_map[num] = hash_map.get(num, 0) + 1
    for num in nums:
        if hash_map[num] > 1:
            return num
    return -1



def find_duplicate2(nums: list[int]) -> int:
    '''
    O(n) timp pt suma
    O(1) spatiu
    '''
    nums_sum = sum(nums)
    nums_len = len(nums)
    s = (nums_len * (nums_len - 1)) // 2;
    return nums_sum - s



def test(func):
    assert func([3, 1, 3, 2]) == 3
    assert func([1, 4, 2, 3, 4, 5, 6, 7]) == 4
    assert func([10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == 10
    assert func([1, 8, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == 8
    assert func([1, 1]) == 1
    assert func([1, 2, 2, 3, 4]) == 2


if __name__ == '__main__':
    test(find_duplicate1)
    test(find_duplicate2)
    rez = find_duplicate2([1, 2, 3, 4, 2])
    print(rez)