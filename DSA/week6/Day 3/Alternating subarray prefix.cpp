#include <bits/stdc++.h>
using namespace std;

int main() {
	// your code goes here
	int t ;
	cin >> t ;
	while(t--){
	    int n ;
	    cin >> n ;
	    vector <int> arr(n) ;
	    int temp = 1 ;
	    
	    for (int x = 0 ; x < n ; x++){
	        cin >> arr[x] ;
	    }
	    /*
	    for(int i = 0 ; i < n-1 ; i++){
	        int ans = 1 ;
	        int temp = 1;
	        if (arr[i] < 0){
	            temp = -1 ;
	        }
	        for (int j = i+1 ; j < n ; ++j){
	            if ((temp == -1 && arr[j] > 0) || (temp == 1 && arr[j] < 0)){
	                ans++ ;
	                temp*=-1 ;
	            }
	            else{
	                break ;
	            }
	        }
	        cout << ans << " " ;
	    }
	    
	    cout << 1 << endl ;
	    */

        vector <int> ans(n) ;


	    for (int x = 0 ; x < n ; x++){
	        cin >> arr[x] ;
	    }
	    
	    ans[n-1] = 1 ;
	    
	    for (int i = n-2 ; i >=0 ; i--){
	        if ((arr[i] > 0 && arr[i+1] < 0 ) || (arr[i] < 0 && arr[i+1] > 0)){
	            
	            ans[i] = 1 + ans[i+1] ;
	        }
	        else {
	            ans[i] = 1 ;
	        }
	        
	    }
	    
	    for(int j : ans){
	        cout << j << " " ;
	    }
	    
	    cout << endl;
	    
	
	    
	}
	

}
