class Solution {
public:
    string frequencySort(string s) {
        
        map< char , int > d ;
        

        string ans = "";

        for (char i : s){
            if (d.find(i) == d.end()){
                d[i] = 1 ;
            }
            else {
                d[i] +=1 ;
            }
        }

        vector<pair<char, int>> vecx(d.begin(), d.end());

        sort(vecx.begin(), vecx.end(), [](pair<char, int> &a, pair<char, int> &b) {
            return a.second > b.second ;
        }) ;

        for (auto &p : vecx) {
            while(p.second--){
                ans +=p.first ;
            }
            
        }
    return ans ;

    }
};