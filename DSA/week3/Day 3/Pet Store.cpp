#include <bits/stdc++.h>
using namespace std;

int main() {
	// your code goes here
	int t ;
	cin >> t  ;
	
	while(t--){
	    int n ;
	    cin >> n ;
	    
	    map <int , int > d ;
	    int flag = 0 ;
	    
	    while(n--){
	        int a ;
	        cin >> a ;
	        if(d.find(a) != d.end()){
	            d[a] += 1 ;
	        }
	        else{
	            d[a] = 1 ;
	        }
	    }
	    
	    for(auto i : d){
	        if (i.second % 2 != 0){
	            flag = 1 ;
	            break ;
	        }
	    }
	    
	    if (flag == 1){
	        cout << "NO" << endl ;
	    }
	    else{
	        cout << "YES" << endl ;
	    }
	}
	

}
