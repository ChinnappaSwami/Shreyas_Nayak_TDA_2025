#include <bits/stdc++.h>
using namespace std;

int main() {
    int t;
    cin >> t;
    while(t--){
        int n;
        cin >> n;
        vector<int> A(n);
        for(int i = 0; i < n; i++){
            cin >> A[i];
        }
        int minlen = n + 1;
        for(int i = 0; i < n; i++){
            set<int> s;
            for(int j = i; j < n; j++){
                s.insert(A[j]);
                if(s.size() > 2){
                    break;
                } else if (s.size() == 2){
                    minlen = min(minlen, j-i+1);
                }
            }
        }
        if(minlen == n + 1){
            cout << -1 << endl;
        } else {
            cout << minlen << endl;
        }
    }
    return 0;
}
