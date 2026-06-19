# Day 6

```
Week-5 Day-6
Merge Sort

https://www.geeksforgeeks.org/dsa/merge-sort/    [merge sort]

Merge sort a important sorting algorithm, which can perform sorting in O(nlogn).

Its basically based on divide and conquer algorithm. First it keeps on dividing the array, and when size remains 1 return back
And while going back you get a new question to solve. Since array of size 1 is obv already sorted. Now the question becomes "given 2 sorted arrays, merge them into another sorted array in O(n)" This can be easily done by keeping 3 pointers for the 2 arrays (i,j) and for the merged array(k) and keep comparing arr1[i] and arr2[j]. Insert the smallest one in arr3[k] and increase the smallest one's index by 1 since it was already inserted. And simply return this array!
Some small optimizations are possible.

So O(n) for this merging process and total log(n)+1 height of the recursion tree. Hence time complexity is O(nlogn).


Questions:-
https://leetcode.com/problems/sort-an-array/    (easy)
https://leetcode.com/problems/merge-k-sorted-lists/ (normal)
https://leetcode.com/problems/number-of-pairs-satisfying-inequality/description/ (normal)


Bonus Resource: https://www.codechef.com/learn/course/college-design-analysis-algorithms/CPDAA03/problems/DAA010
```
