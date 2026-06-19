class Solution {
public:
    bool isBalanced(string& k) {
        stack<char> st;
        for (auto& s : k) {
            if (s == '(' || s == '{' || s == '[') {
                st.push(s);
            } else {
                if (st.empty()){ 
                    return false;
                }
                char top = st.top();
                if ((s == ')' && top != '(') || (s == '}' && top != '{') || (s == ']' && top != '[')) {
                    return false;
                } else {
                    st.pop();
                }
            }
        }
        return st.empty();
    }
};
