// #include <bits/stdc++.h>
// using namespace std;

// int main() {
// 	// your code goes here
// 	int n , m ;
// 	cin >> n >> m ;
// 	vector<long long> poly(n) ; 
	
//     for(int i = 0 ; i < n ; i++){
//         cin >> poly[i] ;
//     }
//     sort(poly.begin(), poly.end());
//     while(m--){
//         int x ;
//         cin >> x ;
        
//         if (find(poly.begin(), poly.end(), x) != poly.end()) {
//             cout << 0 << endl ;
//             continue;
//         }        
//         int ans = 1 ;
        
//         for(int i : poly){
//             ans *= (x - i) ;
//         }
        
//         if (ans > 0 ){
//             cout << "POSITIVE" << endl ;
//         }
//         else if(ans < 0){
//             cout << "NEGATIVE" << endl ;
//         }
        
    

//     }
// }

// #include <bits/stdc++.h>
// using namespace std;

// int main() {
//     int n, m;
//     cin >> n >> m;
//     vector<int> poly(n);
//     for (int i = 0; i < n; i++) {
//         cin >> poly[i];
//     }

//     while (m--) {
//         int x;
//         cin >> x;

//         int sign = 1;
//         for (int root : poly) {
//             int diff = x - root;
//             if (diff == 0) {
//                 sign = 0; // Polynomial is 0 at root
//                 break;
//             } else if (diff < 0) {
//                 sign *= -1;
//             }
//         }

//         if (sign > 0)
//             cout << "POSITIVE" << endl;
//         else if (sign < 0)
//             cout << "NEGATIVE" << endl;
//         else
//             cout << 0 << endl;
//     }
// }


#include <bits/stdc++.h>
using namespace std;

int main() {
	// your code goes here
	int n , m ;
	cin >> n >> m ;
	vector<long long> poly(n) ; 
	
    for(int i = 0 ; i < n ; i++){
        cin >> poly[i] ;
    }
    sort(poly.begin(), poly.end());
    while(m--){
        int x ;
        cin >> x ;
        
        if (binary_search(poly.begin(), poly.end(), x)) {
            cout << 0 << endl ;
            continue;
        }        
        else{
            int id = upper_bound(poly.begin(), poly.end(), x) - poly.begin();
            int temp = n - id;
            if(temp % 2 == 0){
                cout << "POSITIVE" << endl;
            }
            else{
                cout << "NEGATIVE" << endl;
            }
        }
        
    

    }
}
