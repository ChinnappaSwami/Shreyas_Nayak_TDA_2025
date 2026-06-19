# Day 6

```
Week-2 Day-6
Linked Lists

https://www.geeksforgeeks.org/singly-linked-list-tutorial/   [Singly linked lists]
https://www.geeksforgeeks.org/doubly-linked-list/ [doubly linked lists]
https://www.geeksforgeeks.org/dsa/circular-linked-list/ [circular linked lists]

Linked lists, its like a train, or some boxes [data] connected with chains to "link" them together. Unlike arrays which is like a "Bus" where everything is contiguous or continuous or basically the boxes (data) are just next to each other, here the boxes are in different locations connected by some chains.
Linked lists main advantage is the insertion and deletion, you can simply unhook the chain of a box and connect other box in between! The only disadvantage is accessing these boxes as you will only have information about head / the first box, you would have to iterate all the boxes to finally be able to find the box you were looking for! The advantage of arrays is obv the access time by index being just O(1) while linked lists by index or value is O(n).

There are mainly 3 types of linked lists:-
Singly linked lists -> a box stores the data and the location of the next box.
Doubly linked lists -> a box stores the data and the location of the next & the prev box.
Circular linked lists -> as the name suggests its a circular chain, there is no "end" to it the last box will have info for the first box's location.

Linked lists can also be used to implement Queues, Stacks with the advantages being different from arrays.
Don't confuse linked lists and the "list" of python, they are different.

https://www.geeksforgeeks.org/problems/delete-n-nodes-after-m-nodes-of-a-linked-list/1 (easy-normal)
https://leetcode.com/problems/reverse-linked-list/ (easy-normal)
https://leetcode.com/problems/delete-the-middle-node-of-a-linked-list/ (easy-normal)
Bonus: https://leetcode.com/problems/reverse-nodes-in-k-group/ (normal)
```