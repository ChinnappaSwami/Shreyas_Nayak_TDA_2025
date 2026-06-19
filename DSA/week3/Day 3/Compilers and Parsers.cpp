#include <bits/stdc++.h>
using namespace std;

int main() {
	// your code goes here
	int t;
	cin >> t  ;
	
	while(t--){
	    string s ;
	    cin >> s ;
	    int ans = 0 ;
	    int c = 1;
	    
	    
	    if (s[0] == '>'){
	        cout << 0 <<  endl ;
	        continue ;
	        
	    }
	    
	    
	    
	    for (int i = 1 ; i < s.size() ; i++){
	        if (s[i] == '>'){
	            c -= 1 ;
	        }
	        else{
	            c += 1 ;
	        }
	        
	        if (c == 0 ){
	            ans = (i+1) ;
	        }
	        else if (c < 0){
	            break ;
	        }
	    }
	    
	    cout << ans << endl;
	}
	return 0 ;

}
