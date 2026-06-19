#include <bits/stdc++.h>
using namespace std;

struct Cow {
    int a, t, s;
};

int main() {
    int n;
    cin >> n;

    vector<Cow> v(n);
    for (int i = 0; i < n; ++i) {
        cin >> v[i].a >> v[i].t;
        v[i].s = i;
    }

    sort(v.begin(), v.end(), [](Cow x, Cow y) {
        return x.a < y.a;
    });

    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> q;

    long long time = 0;
    int i = 0, ans = 0;

    while (i < n || !q.empty()) {
        while (i < n && v[i].a <= time) {
            q.push({v[i].s, i});
            ++i;
        }

        if (!q.empty()) {
            int idx = q.top().second;
            q.pop();
            ans = max(ans, (int)(time - v[idx].a));
            time += v[idx].t;
        } else {
            time = v[i].a;
            q.push({v[i].s, i});
            ++i;
        }
    }

    cout << ans << '\n';
}
