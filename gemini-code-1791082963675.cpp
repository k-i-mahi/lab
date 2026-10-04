#include <iostream>
#include <vector>

using namespace std;
typedef long long ll;

ll modExp(ll b, ll e, ll m) {
    ll r = 1; b %= m;
    while (e > 0) {
        if (e & 1) r = (__int128)r * b % m;
        b = (__int128)b * b % m;
        e >>= 1;
    }
    return r;
}

ll extGCD(ll a, ll b, ll &x, ll &y) {
    if (b == 0) { x = 1; y = 0; return a; }
    ll x1, y1, g = extGCD(b, a % b, x1, y1);
    x = y1; y = x1 - y1 * (a / b);
    return g;
}

ll modInverse(ll e, ll phi) {
    ll x, y;
    ll g = extGCD(e, phi, x, y);
    if (g != 1) return -1;
    return (x % phi + phi) % phi;
}

ll crackRSA(ll e, ll n) {
    for (ll i = 2; i * i <= n; ++i) {
        if (n % i == 0) {
            ll p = i, q = n / i;
            return modInverse(e, (p - 1) * (q - 1));
        }
    }
    return -1;
}

int main() {
    // 1. Key Generation
    ll p = 61, q = 53, n = p * q, phi = (p - 1) * (q - 1);
    ll e = 3;
    ll x_dummy, y_dummy;
    while (e < phi && extGCD(e, phi, x_dummy, y_dummy) != 1) e += 2;
    ll d = modInverse(e, phi);
    cout << "RSA KeyGen -> Public: (" << e << ", " << n << ") | Private d: " << d << "\n";

    // 2. Encryption & Decryption
    ll m = 89;
    ll c = modExp(m, e, n);
    cout << "Enc: " << c << " | Dec: " << modExp(c, d, n) << "\n";

    // 3. Multiplicative Homomorphic Property (Product Cipher)
    ll m1 = 7, m2 = 11;
    ll c_prod = (modExp(m1, e, n) * modExp(m2, e, n)) % n;
    cout << "Product Cipher Dec: " << modExp(c_prod, d, n) << " (Expected: " << m1 * m2 << ")\n";

    // 4. Brute Force Recovery of d
    cout << "Cracked d via factor search: " << crackRSA(e, n) << "\n";

    // 5. Known-Plaintext Attack: Find all decryption exponents for known (m, c)
    cout << "Matching d candidates for m=" << m << ": ";
    for (ll test_d = 1; test_d < phi; ++test_d) {
        if (modExp(c, test_d, n) == m && (e * test_d) % phi == 1)
            cout << test_d << " ";
    }
    cout << "\n";

    // 6. Digital Signature & Sign-Then-Encrypt
    // Receiver key pair
    ll p2 = 79, q2 = 83, n2 = p2 * q2, phi2 = (p2 - 1) * (q2 - 1);
    ll e2 = 17, d2 = modInverse(e2, phi2);

    ll sig = modExp(m, d, n);                 // Alice signs m with her private d
    ll enc_sig = modExp(sig, e2, n2);         // Alice encrypts signature with Bob's e2
    ll enc_m = modExp(m, e2, n2);             // Alice encrypts message with Bob's e2

    // Bob decrypts both, then verifies Alice's signature with Alice's e
    ll dec_sig = modExp(enc_sig, d2, n2);
    ll dec_m = modExp(enc_m, d2, n2);
    bool valid = (modExp(dec_sig, e, n) == dec_m);
    cout << "Sign-then-Encrypt Verified: " << (valid ? "TRUE" : "FALSE") << "\n";

    return 0;
}