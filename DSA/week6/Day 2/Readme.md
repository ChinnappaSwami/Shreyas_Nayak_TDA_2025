# Day 2

```
Week-6 Day-2
Advanced Maths-2!
Learning Day! no questions.

Extended GCD algorithm - https://cp-algorithms.com/algebra/extended-euclid-algorithm.html 
Modulo Inverse - https://cp-algorithms.com/algebra/module-inverse.html [ study the first method only, study the 2nd method after studying the below 2 theorems]
Euler's Theorem - https://www.geeksforgeeks.org/maths/eulers-theorem
Fermat's Little Theorem - https://www.geeksforgeeks.org/dsa/fermats-little-theorem (special case of eulers theorem)
Bonus -> Linear Diophantine Equation - https://cp-algorithms.com/algebra/linear-diophantine-equation.html


Bezouts Identity states that a solution for ax+by = gcd(a,b) always exists.

Extended GCD algorithm is a way to solve for x,y in the equation:- ax + by = gcd(a,b).
we know gcd(a,b) = gcd(b, a%b)
now say we write this equation in another way for (x1,y1):-
b*x1  + (a%b)*y1 = gcd (b, a%b)
since gcd(a,b) = gcd(b, a%b)
b*x1  + (a%b)*y1 = ax + by

we know a%b = a - floor(a/b)*b 

b*x1 + a*y1 - floor(a/b)*b*y1 = ax + by
a*y1 + b*(x1 - floor(a/b)*y1) = ax + by

hence x = y1
and y = (x1 - floor(a/b)*y1)
now these values can be easily found recursively.


Modulo Inverse:-
Modulo inverse is the solution to the equation:-
ax ≡ 1 mod m
where x is the modulo inverse of a. Modulo Inverse exists only if gcd(a,m) = 1.
we can solve it using extended gcd or using eulers theorem. Lets study the extended gcd method first:-
ax + my = 1   [ since gcd(a,m)=1]
(ax+my) % m = 1%m
(ax) % m = 1%m   [ since my % m = 0]
ax ≡ 1 mod m.
hence ax + my = 1 finding solution for x,y will result into finding inverse of a which is x. and inverse of m which is y.




Eulers theorem states that:-

a^(phi(n)) ≡ 1 (mod n)
where a is any integer coprime with n, n>0, phi(n) is eulers totient.
This formula is based on the fact that if:-
gcd(a,n) = 1, gcd(x,n)=1 then gcd(ax,n) is also = 1
hence let a1,a2,....ak be a set of all numbers less than relatively prime to n, where k = phi(n)
since gcd(a,n) = 1, its obivously one of these numbers.
so a*a1,a*a2,....a*ak is the set of all numbers relatively prime to n. why? due to gcd(a,n)=1, gcd(ai,n)=1, so gcd(a*ai,n)=1.
but a*ai can be greater than n.
hence, all mod n is going to be the same set.
a*a1%n,a*a2%n,....a*ak%n  is the same set as a1,a2,....ak  with just different permutation.
lets find product of this set:-
a*a1*a*a2...a*ak ≡ a1*a2...*ak mod n
a^k * a1*a2*a3....*ak ≡ a1*a2...*ak mod n
since a1*a2*a3....*ak is coprime to n, this has a inverse modulo. Hence we can cancel it from both sides.
a^k ≡ 1 mod n   hence proved.

Fermats Little theorem is a special case of eulers theorem where n is a prime, hence phi(n) = n*(1-1/n) = (n-1)
hence a^(n-1) ≡ 1 mod n
since gcd(a,n)=1 its modulo inverse exists hence:-
a^(n) ≡ a mod n. [ this is fermats little theorem]
i.e. a^n - a is a integer multiple of n, where n is a prime number.


You can study more about solving a more general case ax+by = c  in the  bonus link.[ where a,b,c are integers. This is called as Linear Diophantine equation]
```
