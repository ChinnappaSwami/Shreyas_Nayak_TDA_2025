
void multiply(int A[][100], int B[][100], int C[][100], int N) {
    for(int i = 0; i < N; i++){
        C[i] = {0} ;
    }
    
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            for(int k = 0; k < N; k++){
                C[i][j] += A[i][k]*B[k][j];
            }
        }
    }
}