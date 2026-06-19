class Solution {
public:
    vector<int> asteroidCollision(vector<int>& asteroids) {
        stack<int> st;

        for(int num : asteroids){
            bool destroy = false;
            while(!st.empty() && num < 0 && st.top() > 0){
                if(abs(num) > st.top()){
                    st.pop();
                } else if (abs(num) == st.top()){
                    st.pop();
                    destroy = true;
                    break;
                } else {
                    destroy = true;
                    break;
                }
            } 
            if(!destroy){
                st.push(num);
            }
        }
        vector<int> result;
        while(!st.empty()){
            result.push_back(st.top());
            st.pop();
        }
        reverse(result.begin(), result.end());
        return result;
    }
};