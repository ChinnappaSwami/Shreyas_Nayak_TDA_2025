/*
You throw a dice n times, and every throw produces an outcome between 1 and 6. What is the probability that the sum of outcomes is between a and b?
*/

#include <bits/stdc++.h>
using namespace std;

int n, a, b, t,f;
void rec(int dice, int curr){
    if(dice == 0){
        t++;
        if(curr >= a && curr <= b){
            f++;
        }
        return;        
    }
        for(int no = 1; no <= 6; no++){
            rec(dice-1,curr+no);
        }
}
int main(){
    cin>>n>>a>>b;
    rec(n,0);
    double prob = (double)f/t;
    cout<<prob<<endl;
    return 0;
}