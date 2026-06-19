class Solution {
  public:
    string FirstNonRepeating(string &s) {
        int freq[26] = {0};
        queue<char> q;
        string res = "";

        for (char& ch : s) {
            freq[ch - 'a']++;
            q.push(ch);

        while (!q.empty() && freq[q.front() - 'a'] > 1) {
            q.pop();
        }

        if (!q.empty()) {
            res += q.front();
        } else {
            res += '#';
        }
    }

    return res;
    }
};