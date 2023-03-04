"""
Să se determine al k-lea cel mai mare element al unui șir de numere cu n elemente (k < n).
De ex. al 2-lea cel mai mare element din șirul [7,4,6,3,9,1] este 7.
"""

import heapq
import random


def using_sort(nums: list[int], k: int) -> int:
    '''
    O(n*logn)
    '''
    nums.sort(reverse=True)
    return nums[k - 1]



def min_heap(nums: list[int], k: int) -> int:
    '''
    Priority queue / min heap
    O(n*logk) timp
    O(k) spatiu
    '''
    heap = []
    for num in nums:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)
    return heap[0]



def quick_select(nums: list[int], k: int) -> int:
    '''
    O(n) average time da worst case O(n^2)
    O(logn) spatiu pt stive
    '''
    return quick_sel(nums, 0, len(nums) - 1, len(nums) - k)

def quick_sel(nums: list[int], left: int, right: int, k: int) -> int:
    if left == right:
        return nums[left]
    
    pIndex = random.randint(left, right - 1)
    pIndex = partition(nums, left, right, pIndex)
    
    if pIndex == k:
        return nums[k]
    elif pIndex < k:
        return quick_sel(nums, pIndex + 1, right, k)
    else:
        return quick_sel(nums, left, pIndex - 1, k)
    
def partition(nums: list[int], left: int, right: int, pIndex: int) -> int:
    pivot = nums[pIndex]
    nums[right], nums[pIndex] = nums[pIndex], nums[right]
    pIndex = left
    
    for i in range(left, right + 1):
        if nums[i] <= pivot:
            nums[i], nums[pIndex] = nums[pIndex], nums[i]
            pIndex += 1
    
    return pIndex - 1
        


def test(func):
    assert func([7, 4, 6, 3, 9, 1], 2) == 7
    assert func([3, 2, 1, 5, 6, 4], 2) == 5
    assert func([3, 2, 3, 1, 2, 4, 5, 5, 6], 4) == 4
    assert func([3, 2, 3, 1, 2, 4, 5, 5, 6], 1) == 6
    assert func([3, 2, 3, 1, 2, 4, 5, 5, 6], 2) == 5
    assert func([3, 2, 3, 1, 2, 4, 5, 5, 6], 3) == 5


if __name__ == '__main__':
    test(using_sort)
    test(min_heap)
    test(quick_select)
    rez = quick_select([7, 4, 6, 3, 9, 1], 2)
    print(rez)