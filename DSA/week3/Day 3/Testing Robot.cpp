#include <bits/stdc++.h>
using namespace std;

int main() {
	// your code goes here
	int t ;
	cin >> t ;
	
	while(t--){
	    int n , x ;
	    cin >> n >> x ;
	    
	    unordered_set<int> s ;
	    
	    s.insert(x) ;
	    string si ;
	    cin >> si  ;
	    
	    for(char i : si){
	        
	        if(i == 'R'){
	            x += 1 ;
	        }
	        else{
	            x -= 1 ;
	        }
	        s.insert(x) ;
	    }
	    
	    cout << s.size() << endl;
	    
	}

}
