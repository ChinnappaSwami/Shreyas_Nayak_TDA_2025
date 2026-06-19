class Solution {
public:
    vector<int> intersection(vector<int>& nums1, vector<int>& nums2) {

        map<int, bool> Map;
        vector<int> result;

        for(int n : nums1){
            Map[n] = true;
        }

        for(int n : nums2){
            if(Map.find(n) != Map.end() && Map[n]){
                result.push_back(n);
                Map[n] = false;
            }
        }
        return result;
    }
};
    
