# Day 6

```
Week-3 day-6
Binary Search!
This is a very important algorithm, as it can be used in a lot of question where it may not even seem like a binary search problem!
https://www.geeksforgeeks.org/dsa/binary-search/ [for learning binary search]
https://www.geeksforgeeks.org/dsa/searching-algorithms/ [library implementation of binary search]

We all are familiar with the problem, given an array of integers find a element x and return its first index. You can easily solve it using linear search where u start from index 0 and go to the end and check whose value matches.

Now can we make it faster? Yes. But only for "sorted arrays" [meaning in ascending or desceing order and in some special orders as well :p ]
Lets assume a ascending ordered arrat of integers, say [1,2,5,7,10], find index of x=7.
Binary search goes like its name it finds the halve which will contain that element if it exists. Binary search checks the middle element of the array, lets call it mid. if x the element we are finding is greater than mid its obv on the right half of the array. If x is lesser than mid then its obv on the left half of the array! Now repeat this process on that half of the array untill u get x equal to mid or the size of subarray that may containt it is now only 1.
So in the example
[1,2,5,7,10] [left=0, right=4] so mid index = (4+0)/2 so mid index is 4. check 5<7 so hence right side of the array so left = mid index +1 so left = 3.
[1,2,5,7,10] [left=3, right=4] mid index=(3+4)/2, so mid index=3.and 7==7 hence we found our element!
This methods time complexity is O(logn).

This can be used in other situations where it may not look like a "search" problem as well! Like some questions may require you to find smth, you know what that smth has a lower and upper bounds [basically u know ur answer lies between that range] and you also know to check if say x is a solution to that question or not in O(k) time complexity. And you know if that x is not the answer then u can deduce that one of the halfs will never contain the answer .Then just perform a binary search on that range and check each mid element if tats the answer. With time complexity O(klogn)

U also got tertiary search which divides the array into 3 halves and then decides on one half which may or may not contain it. Answer in the group will it be fastrr than binary or not?

https://leetcode.com/problems/binary-search/ (easy)
https://leetcode.com/problems/sqrtx/description/ (easy)
https://leetcode.com/problems/search-in-rotated-sorted-array/description/ (normal)
Bonus -> https://leetcode.com/problems/search-a-2d-matrix/description/ (normal)
Bonus -> https://codeforces.com/problemset/problem/1985/F (good)
```