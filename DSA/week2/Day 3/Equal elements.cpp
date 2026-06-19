#include <bits/stdc++.h>
using namespace std;

int main() {
    int T;
    cin >> T;
    while (T--) {
        int N;
        cin >> N;
        vector<int> A(N);
        unordered_map<int, int> freq;

        int maxFreq = 0;
        for (int i = 0; i < N; ++i) {
            cin >> A[i];
            freq[A[i]]++;
            maxFreq = max(maxFreq, freq[A[i]]);
        }
        cout << (N - maxFreq) << endl;
    }
    return 0;
}
