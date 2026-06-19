/*
Your task is to calculate the number of bit strings of length n.
For example, if n=3, the correct answer is 8, because the possible bit strings are 000, 001, 010, 011, 100, 101, 110, and 111.
*/

#include<bits/stdc++.h>

using namespace std;

int main(){
    
    int n ;
    cin >> n ;
    
    int m = 1e9 + 7 ;
    int ans = 1;
    
    for (int i = 0 ; i < n ; i++){
        ans = ans*2 % m ;
    }
    
    cout << ans ;
    
    return 0 ;
    
}