#include <bits/stdc++.h>
using namespace std;

int main() {
	// your code goes here
	int t ;
	cin >> t ;
	
	while(t--){
	    int n ;
	    cin >> n ;
	    
	    unordered_map<int , long long > d ;
	    
	    for(int i = 0 ; i< n ; i++){
	        int a ;
	        cin >> a  ;
	        d[a]++ ;
	    }
	    
	    long long p = 0 ;
	    
	    for(auto i : d){
	        if (i.second > 1 ){
	            p += i.second*(i.second - 1) / 2 ;
	        }
	    }
	    
	    cout << p << endl ;
	}
    return 0 ;
    
}
