class Solution {
public:
    int subarraysDivByK(vector<int>& nums, int k) {
       unordered_map<int, int> freq;
       freq[0] = 1;

       int sum = 0,count = 0;

       for(int num : nums){
            sum += num;
            int mod = sum % k;
            if(mod < 0){
                mod += k;
            }

            if(freq.find(mod) != freq.end()){
                count += freq[mod];
            }
            freq[mod]++;
       }
       return count;
    }
};
