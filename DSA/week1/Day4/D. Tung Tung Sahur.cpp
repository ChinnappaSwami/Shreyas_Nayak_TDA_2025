#include <bits/stdc++.h>
using namespace std;

bool check(string p, string s) {
    int i = 0, j = 0;
    while (i < p.size() && j < s.size()) {
        if (p[i] != s[j]) {
            return false;
        }
        char c = p[i];
        int count_p = 0;
        while (i < p.size() && p[i] == c) {
            i++;
            count_p++;
        }
        int count_s = 0;
        while (j < s.size() && s[j] == c) {
            j++;
            count_s++;
        }
        if (count_s < count_p || count_s > 2 * count_p) {
            return false;
        }
    }
    return i == p.size() && j == s.size();
}

int main() {
    int t; 
    cin >> t;
    while (t--) {
        string p, s;
        cin >> p >> s;
        cout << (check(p, s) ? "YES" : "NO") << endl;
    }
    return 0;
}