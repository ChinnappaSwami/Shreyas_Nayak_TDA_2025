/*
The Tower of Hanoi game consists of three stacks (left, middle and right) and n round disks of different sizes. Initially, the left stack has all the disks, in increasing order of size from top to bottom.
The goal is to move all the disks to the right stack using the middle stack. On each move you can move the uppermost disk from a stack to another stack. In addition, it is not allowed to place a larger disk on a smaller disk.
Your task is to find a solution that minimizes the number of moves.
*/


#include<bits/stdc++.h>
using namespace std;

int leftPeg[20], middlePeg[20], rightPeg[20];
int leftTop = -1, middleTop = -1, rightTop = -1;

void placeDisk(int peg[], int &top, int disk){
    top++;
    peg[top] = disk;
}

int removeDisk(int peg[], int &top){
    int disk = peg[top];
    top--;
    return disk;
}

int checkDisk(int peg[], int top){
    if(top == -1){
        return 1e4;
    }
    return peg[top];
}

void moveDisk(int fromPeg[], int &fromTop, int toPeg[], int &toTop, int fromNum, int toNum){
    int topFrom = checkDisk(fromPeg, fromTop);
    int topTo = checkDisk(toPeg, toTop);

    if(topFrom > topTo){
        int disk = removeDisk(toPeg, toTop);
        placeDisk(fromPeg, fromTop, disk);
        cout << toNum << " " << fromNum << endl;
    } else {
        int disk = removeDisk(fromPeg, fromTop);
        placeDisk(toPeg, toTop, disk);
        cout << fromNum << " " << toNum << endl;
    }
}

int main() {
    int n;
    cin >> n;
    int moves = (1 << n) - 1;
    cout << moves << endl;

    for(int i = n; i >= 1; i--){
        placeDisk(leftPeg, leftTop, i);
    }

    for(int i = 1; i <= moves; i++){
        if(n % 2 == 0){
            if(i % 3 == 1){
                moveDisk(leftPeg, leftTop, middlePeg, middleTop, 1, 2);
            } else if(i % 3 == 2){
                moveDisk(leftPeg, leftTop, rightPeg, rightTop, 1, 3);
            } else {
                moveDisk(middlePeg, middleTop, rightPeg, rightTop, 2, 3);
            }
        } else {
            if(i % 3 == 1){
                moveDisk(leftPeg, leftTop, rightPeg, rightTop, 1, 3);
            } else if(i % 3 == 2){
                moveDisk(leftPeg, leftTop, middlePeg, middleTop, 1, 2);
            } else {
                moveDisk(middlePeg, middleTop, rightPeg, rightTop, 2, 3);
            }
        }
    }
    return 0;
}


