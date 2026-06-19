#include <bits/stdc++.h>
using namespace std;

int main() {
    int t;
    cin >> t;
    while(t--){
        int n,m;
        cin>>n>>m;
        int row = n % 2;
        int column = m % 2;
        int result = row * m + column * n - row * column;
        cout << result << endl;
    }
    return 0;
}
