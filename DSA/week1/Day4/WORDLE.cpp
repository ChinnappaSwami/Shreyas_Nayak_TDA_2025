#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    cin>>n;
    while(n --){
        string a,b;
        cin>>a>>b;
        string M = "";
        for(int i = 0; i < 5;i++){
            if (a[i] == b[i]) {
                M += 'G';
            } else {
                M += 'B';
            }
        }
        cout<< M <<"\n";
    }
    return 0;
}