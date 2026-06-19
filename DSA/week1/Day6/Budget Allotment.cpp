#include <bits/stdc++.h>
using namespace std;

int budgetAllotment(int n, int x, vector<int> &A){
    
}

int main() {
    int t;
    cin >> t;
    while(t--){
        int n, x;
        cin >> n >> x;
        vector<int> A(n);
        for(int i = 0; i < n; i++){
            cin >> A[i];
        }   
        sort(A.rbegin(), A.rend());
    	int count = 0;
    	long long int excess = 0;
    
    	for(int i = 0; i < n; i++){
        	if(A[i] >= x){ 
         	   count++;
         	   excess += (A[i] - x);
        	} else {
        	    int needed = x - A[i];
        	    if(excess >= needed){
        	        excess -= needed;
         	       count++;
         	   } else {
         	       break;
         	   }
        	}
    	}
    	cout << count << endl ;
    }
    return 0;
}