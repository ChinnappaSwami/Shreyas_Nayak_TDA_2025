#include <bits/stdc++.h>
using namespace std;

int main() {
    int T;
    cin >> T;
    while (T--) {
        string s;
        cin >> s;
        stack<char> st;
        int maxLength = 0;

        for (int i = 0; i < s.length(); ++i) {
            if (s[i] == '<') {
                st.push('<');
            } else {
                if (st.empty()) {
                    break;
                }
                st.pop();
                if (st.empty()) {
                    maxLength = i + 1;
                }
            }
        }

        cout << maxLength << endl;
    }
    return 0;
}
