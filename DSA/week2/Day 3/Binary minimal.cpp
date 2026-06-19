#include <bits/stdc++.h>
using namespace std;

int main() {
	// your code goes here
    int t;
    cin>>t;
    while(t--){
      int n,k,c;
      cin>>n>>k;
      string s;
      cin>>s;
        if(k){
         c=count(s.begin(),s.end(),'1'); 
         if(c<=k){
             string res((n-k),'0');
             cout<<res<<endl;
         }
         else{
            for(int i= 0; i<n&& k>0; i++) {
            if (s[i] == '1') {
                s[i] = '0';
                k--;}}
                cout<<s<<endl;

         }
         
         
            }
         else cout<<s<<endl;
        }
       
    }

