#include<bits/stdc++.h>
using namespace std;

int main() {
    string s;
    cin >> s;
    int n = s.size();

    vector<int> freq(26, 0);
    for (char c : s)
        freq[c - 'A']++;

    for (int f : freq) {
        if (f > (n + 1) / 2) {
            cout << -1 << endl;
            return 0;
        }
    }

    string result = "";
    char prev = '.';

    for (int i = 0; i < n; ++i) {
        bool placed = false;
        for (int c = 0; c < 26; ++c) {
            if (freq[c] == 0) continue;
            char curr = 'A' + c;
            if (curr == prev) continue;

            freq[c]--;
            int remaining = n - i - 1;

            int maxFreq = 0;
            for (int j = 0; j < 26; ++j)
                maxFreq = max(maxFreq, freq[j]);

            if (maxFreq <= (remaining + 1) / 2) {
                result += curr;
                prev = curr;
                placed = true;
                break;
            }

            freq[c]++;
        }

        if (!placed) {
            cout << -1 << endl;
            return 0;
        }
    }

    cout << result << endl;
    return 0;
}