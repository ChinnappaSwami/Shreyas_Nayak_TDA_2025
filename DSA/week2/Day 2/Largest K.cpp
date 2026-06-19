#include <bits/stdc++.h>
using namespace std;

int main() {
    int T;
    cin >> T;
    while(T--){
        int n;
        cin >> n;
        vector<int> a(n);
        unordered_map<int, int> freq;
        for(int i = 0; i < n; i++){
            cin >> a[i];
            freq[a[i]]++;
        }
        vector<int> values;
        for(auto& p : freq){
            values.push_back(p.second);
        }
        sort(values.rbegin(), values.rend());
        int k_max = 0, sum = 0;
        for(int d = 1; d <= values.size(); d++){
            sum += values[d-1];
            int k = sum - (sum % d);
            k_max = max(k_max, k); 
        }
        cout<< k_max << endl;
    }
    return 0;
}
