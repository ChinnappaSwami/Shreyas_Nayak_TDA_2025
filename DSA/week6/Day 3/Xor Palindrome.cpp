#include <bits/stdc++.h>
using namespace std;

int main() {
	// your code goes here
    int t;
    cin >> t;
    while (t--) {
        /*
        int n , number;
        cin >> n >> number;
        string num = to_string(number); 
        //cout << num << endl; 
        int ans = 0;
        int ones =  0, zeroes = 0;
        for (int i = 0; i < n/2 ; i++){
            if ( num[i] != num[n-i-1] ){
                if (num[i] == '1'){
                    ones++;
                }
                else{
                    zeroes++;
                    //cout << "-----" << i << "------";
                }
            }
        }
        ans = int(ones/2) + ones%2 + int(zeroes/2) + zeroes%2;
        ans = (ones+ zeroes + 1) /2 ;
        cout  << ans  << endl;
        
        
        
        
        */
       
    int n;
    cin >> n;
    string s;
    cin >> s;

    int mismatch_pairs = 0;

    for (int i = 0; i < n / 2; ++i) {

        if (s[i] != s[n - 1 - i]) {
            mismatch_pairs++;
        }
    }


    int operations = (mismatch_pairs + 1) / 2;
    
    cout << operations << endl;

    }
}