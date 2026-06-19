class Solution {
public:
    bool uniqueOccurrences(vector<int>& arr) {
        unordered_map<int, int> freq;
        for (int num : arr) {
            freq[num]++;
        }

        unordered_set<int> sadge;
        for (auto& entry : freq) {
            if (sadge.find(entry.second) != sadge.end()){
                return false; 
            }
            sadge.insert(entry.second);
        }

        return true;
    }
};
