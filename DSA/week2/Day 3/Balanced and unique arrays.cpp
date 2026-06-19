#include <bits/stdc++.h>
using namespace std;

int main() {
    int t;
    cin >> t;
    while(t--){
        int N;
        cin >> N;
        if(N % 4 != 0){
            cout << "NO" << endl;
            continue;
        }
        cout << "YES" << endl;
        int q = N / 4;
        vector<int> A, B;
        for (int i = 1; i <= q; ++i)
            A.push_back(i);
        for (int i = N - q + 1; i <= N; ++i)
            A.push_back(i);
        for (int i = q + 1; i <= N - q; ++i)
            B.push_back(i);
        for (int x : A) cout << x << " ";
        cout << endl;
        for (int x : B) cout << x << " ";
        cout << endl;
    }
    return 0;
}
