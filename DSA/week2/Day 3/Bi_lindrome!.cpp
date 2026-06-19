#include <bits/stdc++.h>
using namespace std;

int main() {
	// your code goes here
    int t ;
    cin >> t ;
    
    while(t--){
        int a ;
        string s ;
        map <char , int > dic ;
        
        cin >> a >>  s ;
        
        int c = 0 ;
        
        for (char i : s){
            if (dic.find(i) == dic.end()){
                dic[i] = 1 ;
            }
            else {
                dic[i] ++ ;
                
            }
        }
        for (auto j :dic ){
            if (j.second > 1 ){
                c +=2 ;
            }
        }
        
        if (c == 0){
            cout << -1 <<endl;
        }
        else {
            cout << (a-2) <<endl;
        }
    }
    return 0 ;
}