# Day 4

```
Week-4 Day-4

DFS & BFS

https://www.geeksforgeeks.org/depth-first-search-or-dfs-for-a-graph/ [DFS]
https://www.geeksforgeeks.org/breadth-first-search-or-bfs-for-a-graph/ [BFS]
https://www.geeksforgeeks.org/difference-between-bfs-and-dfs/ [BFS vs DFS]

DFS - depth first search, i.e. suppose ur exploring ur city, ur at a round about, u choose one path and decided to continue on that path until the road ends, then u go back to the point where another branch of the road meets and u explore that branch and so on. i.e. u just keep going on one path until it ends or it loops back to the same place.

BFS - Breadth first search, i.e. in the city example one road leads to a famous Chinese restaurant and going further in that route it leads to a car showroom [u like cars], and another road leads to City Museum, You wanna visit all these 3 places, but using BFS you would first visit either city museum or the Chinese restaurant, then return back and then visit the car showroom.

Example:-

A-B-C-F   suppose u got a graph like this on the side [ note: F is connected with D ]
    |     /
    |   /
   D/
   |
   |
   E 
Using DFS here are possible solutions [ many ways are actually possible different implementation leads to different results but the main idea remains same]
A->B->C->F->D->E
A->B->D->E->F->C
A->B->D->F->C->E

using BFS here are possible solutions
A->B->C->D->F->E
A->B->D->C->E->F

with many more implementations more results are possible, these above are the main ones.

https://leetcode.com/problems/number-of-islands/description/ [Normal]
https://leetcode.com/problems/flood-fill/description/ [Easy]
Bonus -> https://codeforces.com/contest/1979/problem/E [Not exactly a graph problem, but a very good question, based on different concepts.]
Bonus-> https://leetcode.com/problems/cheapest-flights-within-k-stops/description/ [ i didn't explain Dijkstra's yet, but this can be implemented using just BFS]
```
