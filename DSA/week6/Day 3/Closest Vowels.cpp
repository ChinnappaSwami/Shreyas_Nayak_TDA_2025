#include <bits/stdc++.h>
using namespace std;

int main() {
	// your code goes here
	int t;
	cin >> t;
	string m = "crgl" ;
	while (t--) {
	    int n;
	    string s;
	    cin >> n >> s;
	    long long c = 1;
	    for (char i: s) {
	        if (m.find(i) != string::npos) {
	            c = (c*2) % 1000000007;
	        }
	    }
        cout << c << endl;
        
	}
    
}


