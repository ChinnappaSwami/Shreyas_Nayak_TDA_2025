class Solution {
  public:
    int gcd(int a, int b){
        while(b != 0){
            int temp = b;
            b = a % b;
            a = temp;
        }
        return a;
    }
    
    int gcd(int n, vector<int> arr) {
        int result = arr[0];
        for(int i = 0; i < arr.size(); i++){
            result = gcd(result, arr[i]);
        }
        return result;
    }
};