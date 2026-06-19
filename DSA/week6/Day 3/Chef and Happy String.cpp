#include <bits/stdc++.h>
using namespace std;

int main() {
    int t;
    cin>>t;
    while(t--){
        string s;
        cin>>s;
        string v = "aeiou";
        int flag = 0 ;
        for (int i=0; i< s.size()-2 ; i++){
            if ((v.find(s[i]) != string::npos) && (v.find(s[i+1]) != string::npos) && (v.find(s[i+2]) != string::npos)){
                cout << "Happy" << endl;
                flag = 1;
                break;
            } 
        }
        if (!flag){
            cout << "Sad" << endl;
        }
        
    }

}