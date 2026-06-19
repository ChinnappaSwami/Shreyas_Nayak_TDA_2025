class Solution {
  public:
    queue<int> reverseFirstK(queue<int> q, int k) {
        if(q.size() < k){
            return q;
        } 
        stack<int> s;
        
        for(int i = 0; i < k; i++){
            s.push(q.front());
            q.pop();
        }
        
        while(!s.empty()){
            q.push(s.top());
            s.pop();
        }
        
        for(int i = 0; i < q.size() - k; i++){
            q.push(q.front());
            q.pop();
        }
        
        return q;
    }
};