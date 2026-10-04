#include <iostream>

using namespace std;
typedef long long ll;

ll extGCD(ll a, ll b, ll &x, ll &y) {
    if (b == 0) { x = 1; y = 0; return a; }
    ll x1, y1, g = extGCD(b, a % b, x1, y1);
    x = y1; y = x1 - y1 * (a / b);
    return g;
}

ll modInv(ll a, ll m) {
    ll x, y;
    extGCD(a, m, x, y);
    return (x % m + m) % m;
}

// Curve parameters: y^2 = x^3 + a*x + b mod p
const ll p = 97, a = 2, b = 3;

struct Point {
    ll x, y;
    bool inf;
};

Point add(Point P, Point Q) {
    if (P.inf) return Q;
    if (Q.inf) return P;
    if (P.x == Q.x && (P.y + Q.y) % p == 0) return {0, 0, true};

    ll lambda;
    if (P.x == Q.x && P.y == Q.y) {
        lambda = (3 * P.x % p * P.x % p + a) % p * modInv(2 * P.y % p, p) % p;
    } else {
        ll num = (Q.y - P.y + p) % p;
        ll den = (Q.x - P.x + p) % p;
        lambda = num * modInv(den, p) % p;
    }
    ll xr = (lambda * lambda - P.x - Q.x) % p;
    if (xr < 0) xr += p;
    ll yr = (lambda * (P.x - xr) - P.y) % p;
    if (yr < 0) yr += p;
    return {xr, yr, false};
}

Point multiply(ll k, Point P) {
    Point R = {0, 0, true};
    while (k > 0) {
        if (k & 1) R = add(R, P);
        P = add(P, P);
        k >>= 1;
    }
    return R;
}

int main() {
    Point G = {3, 6, false};  // Generator point on curve y^2 = x^3 + 2x + 3 mod 97
    ll d = 5;                 // Private scalar key
    Point Q = multiply(d, G); // Public key point Q = d * G

    cout << "ECC KeyGen -> Public Key Q: (" << Q.x << ", " << Q.y << ")\n";

    // 1. Encryption and Decryption (Message M is a point on curve)
    Point M = {10, 24, false};
    ll k = 3; // Ephemeral key
    Point C1 = multiply(k, G);
    Point C2 = add(M, multiply(k, Q));

    // Decryption: M = C2 - d * C1 = C2 + (d * C1_negated)
    Point S = multiply(d, C1);
    Point S_neg = {S.x, (p - S.y) % p, S.inf};
    Point recovered = add(C2, S_neg);
    cout << "Decrypted Point: (" << recovered.x << ", " << recovered.y << ")\n";

    // 2. Homomorphic Addition: E(M1) + E(M2) = E(M1 + M2)
    Point M1 = M, M2 = {13, 20, false};
    Point C1_a = multiply(2, G), C2_a = add(M1, multiply(2, Q));
    Point C1_b = multiply(4, G), C2_b = add(M2, multiply(4, Q));

    Point C1_sum = add(C1_a, C1_b);
    Point C2_sum = add(C2_a, C2_b);

    Point S_sum = multiply(d, C1_sum);
    Point recovered_sum = add(C2_sum, {S_sum.x, (p - S_sum.y) % p, S_sum.inf});
    Point expected_sum = add(M1, M2);

    cout << "Homomorphic Point Sum Decrypted: (" << recovered_sum.x << ", " << recovered_sum.y << ")\n";
    cout << "Direct Point Sum (Expected):    (" << expected_sum.x << ", " << expected_sum.y << ")\n";

    return 0;
}