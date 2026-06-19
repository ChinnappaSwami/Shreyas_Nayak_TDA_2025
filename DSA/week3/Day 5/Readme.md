# Day 5

```
Week-3 Day-5
Tabulation! [Dynamic Programming]
I hope most of you have caught up after giving yall a free day :)
https://www.geeksforgeeks.org/dsa/introduction-to-dynamic-programming-data-structures-and-algorithm-tutorials/  [intro to dp]
https://www.geeksforgeeks.org/dsa/tabulation-vs-memoization/ [tabulation vs memorisation]
https://www.geeksforgeeks.org/dsa/solve-dynamic-programming-problem/ [How to identify if its a DP problem?]

Tabulation, is basically a bottom up approach, what does it even mean?? it basically means its not a recursion, its a iteration in a table starting from 0th (starting) index. Now this table can be 1d, 2d, 3d and with even more dimensions for some very complex dp problems. How is it better than memorisation? coz it uses a iterative approach to solve the problem which takes less memory and faster than recursion, but why would memorisation wwill ever be required? memorisation is in some questions faster due to not requiring to calculate some subproblems that are not required for the final answer [table filled only when required], but in tabulation you solve every subproblem from the bottom once. Memorisation is also easier/natural to think of than a tabulation solution [ at least i feel that way ]. 

How to tabulate?
-> Intialise
-> Formulate [ find a formula for F(n) in terms of F(x), where x<n. ] [this formulate is usually easier in memorisation than in tabulation, coz you start from index 0 in tabulation]
-> Code it out :p [ usually coding part is the easy part of DP problems, the hardest part is to think of the formula, only practicing can improve this. ]


https://www.geeksforgeeks.org/problems/count-ways-to-reach-the-nth-stair-1587115620/1 (Easy) [solve using tabulation ]
https://leetcode.com/problems/jump-game-ii/ (Normal)
https://leetcode.com/problems/unique-paths-ii (Normal)
Bonus -> https://codeforces.com/contest/1282/problem/B2 [Normal-Good]
Bonus -> https://www.geeksforgeeks.org/problems/0-1-knapsack-problem0945/1 [Normal-Good] [ Knapsack 0/1 ]
```
