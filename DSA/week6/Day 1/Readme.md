# Day 1

```
Week-6 Day-1  [Advanced Algorithms Week]
Advanced Maths-1!
Today there is just Learning part, no questions. [ DOMAIN EXPANSION!! ]

Euclids GCD algorithm - https://cp-algorithms.com/algebra/euclid-algorithm.html
Modulo Arithmetic - https://www.geeksforgeeks.org/engineering-mathematics/modular-arithmetic/  [Dont Study Modulo Inverse, We will study it in Advanced Maths-2]
Prime Sieve - https://www.geeksforgeeks.org/dsa/sieve-of-eratosthenes/
Eulers Totient - https://www.geeksforgeeks.org/dsa/eulers-totient-function/


Most might know what Euclids GCD algorithm which is gcd(a,b) = a (if b==0) OR = gcd(b,a mod b) (otherwise)
But can you understand why? Read the first link to know exactly why it happens, if u dont understand it please dm me, ill try explaining it to you. Some questions require the clear understand of this to solve some gcd questions easily with some simplifications.

Modulo Arithmetic is very important as some answers can be really really large, so the question might say "hey can you give me the answer modulo 10^9+7". Now obviously since the number can be so large it just cant be stored easily, hence while performing a operation u need to use modulo's properties, so that these values can be stored easily without affecting our answer. Here are some of the properties:-
1) (a+b) % M = ((a%M) + (b%M)) % M  [This operation helps us a lot, why? we dont have to store a+b, we have to store (a%M) + (b%M) which is obv less than a+b. Hence it wont cause any memory overflow. [ unless 2*M-2 > maximum range of that data it wont overflow ]
2) (a-b) % M = ((a%M) - (b%M)) % M
3) (a*b) % M = ((a%M) * (b%M)) % M
These 3 are the basic properties of modulo. You may wonder, why i didn't write for division, well for that we need to understand Modulo Inverse, which we will study tmrw when we have studied about extended gcd algorithm.

Prime Sieve is a very important algorithm to find all from primes from 1 to N, in O(N*log(logN)). It basically works on the phenomenon that if u found a prime then all numbers wwhwo are divisible by that prime are not prime :). Its like this:- we know 2 is a prime, then all even numbers except 2 are not prime. it uses this idea to find all primes in 1 to N. Read the 3rd link to study more on it.

Eulers totient is also again a important mathematical function that helps us in Hashing and modulo inverses [we will see how later], but first lets define it
Eulers totient of a number X is the count of all numbers from 1 to X that is relatively prime to X [i.e. gcd(i,X)=1, where 1<=i<=X]. Read the 4th link to know more about it.

Bonus Resource:- https://www.youtube.com/watch?v=sD0NjbwqlYw [ Riemann Zeta Function ] (No need to study this, study it if u like maths like me :D )
```
