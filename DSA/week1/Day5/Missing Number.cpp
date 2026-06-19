class Solution {
public:
    int missingNumber(vector<int>& nums) {
        map<int, bool> Map;
        for(int n : nums){
            Map[n] = true;
        }
        for(int i = 0; i <= nums.size(); i++){
            if(Map.find(i) == Map.end()){
                return i;
            }
        }
        return -1;
    }
};