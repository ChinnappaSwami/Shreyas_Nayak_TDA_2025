void getNumberPattern(int n) {
    // Write your code here.

    int s = 2*n - 1 ;

    vector<vector<int>> mat(s, vector<int>(s,0));



    for (int x = 0 ; x < n ; x++){
        int t = n - x;
        for (int i = x ; i < s-x ; i++){
            mat[s-x-1][i] = t ;
            mat[i][s-x-1] = t ;
            mat[x][i] = t ;
            mat[i][x] = t ;
        }
    }

    for (int i = 0; i < s; i++) {
        for (int j = 0; j < s; j++) {
            cout << mat[i][j];
        }
        cout << endl;
    }
}