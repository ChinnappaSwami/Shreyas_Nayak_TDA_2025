class Solution {
  public:
    void printTriangle(int n) {
        // code here
        int  i , j ;
        
        for (i = 0 ; i <= n ; i++){
            for (j = 1 ; j < i+1 ; j++){
                cout << j << " ";
            }
            cout << "\n" ;
        }
    }
};