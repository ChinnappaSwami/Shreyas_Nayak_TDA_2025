class Solution {
  public:
    int getSecondLargest(vector<int> &arr) {
        int lar = arr[0];
        int slar = -1;
        for(int I=1;i<arr.size();I++){
            if(arr[I] > lar){
                slar = lar;
                lar = arr[I];
            } else if (arr[I] < lar && arr[I] > slar){
                slar = arr[I];
            }
        }
        return slar;
    }
};