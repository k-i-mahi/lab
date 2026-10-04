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

ll modInverse(ll a, ll m) {
    ll x, y;
    ll g = extGCD(a, m, x, y);
    if (g != 1) return -1;
    return (x % m + m) % m;
}

// Find prime factors of n
vector<ll> getPrimeFactors(ll n) {
    vector<ll> factors;
    for (ll i = 2; i * i <= n; ++i) {
        if (n % i == 0) {
            factors.push_back(i);
            while (n % i == 0) n /= i;
        }
    }
    if (n > 1) factors.push_back(n);
    return factors;
}

// Compute primitive root generator g modulo prime p
ll findPrimitiveRoot(ll p) {
    ll phi = p - 1;
    vector<ll> factors = getPrimeFactors(phi);
    for (ll g = 2; g < p; ++g) {
        bool ok = true;
        for (ll q : factors) {
            if (modExp(g, phi / q, p) == 1) { ok = false; break; }
        }
        if (ok) return g;
    }
    return -1;
}

struct Ciphertext { ll c1, c2; };

Ciphertext elgamalEnc(ll m, ll p, ll g, ll h, ll k) {
    return { modExp(g, k, p), (m * modExp(h, k, p)) % p };
}

ll elgamalDec(Ciphertext c, ll p, ll x) {
    ll s = modExp(c.c1, x, p);
    return (c.c2 * modInverse(s, p)) % p;
}

int main() {
    // 1. Dynamic Key Generation
    ll p = 467;
    ll g = findPrimitiveRoot(p);
    ll x = 127;                // Private key
    ll h = modExp(g, x, p);    // Public component
    cout << "ElGamal KeyGen -> Prime p: " << p << " | Gen g: " << g << " | Public h: " << h << "\n";

    // 2. Encryption / Decryption
    ll m = 89, k = 105;
    Ciphertext ct = elgamalEnc(m, p, g, h, k);
    cout << "Decrypted: " << elgamalDec(ct, p, x) << "\n";

    // 3. Multiplicative Homomorphism: E(m1) * E(m2) = E(m1 * m2 mod p)
    ll m1 = 6, m2 = 7;
    Ciphertext ct1 = elgamalEnc(m1, p, g, h, 21);
    Ciphertext ct2 = elgamalEnc(m2, p, g, h, 33);
    Ciphertext ct_mult = { (ct1.c1 * ct2.c1) % p, (ct1.c2 * ct2.c2) % p };
    cout << "Homomorphic Mult Dec: " << elgamalDec(ct_mult, p, x) << " (Expected: " << (m1 * m2) % p << ")\n";

    // 4. Re-randomization with new k'
    ll k_prime = 17;
    Ciphertext ct_rerand = { (ct.c1 * modExp(g, k_prime, p)) % p, (ct.c2 * modExp(h, k_prime, p)) % p };
    cout << "Re-randomized Dec: " << elgamalDec(ct_rerand, p, x) << "\n";

    // 5. ElGamal Digital Signature
    ll m_sig = 55, k_sig = 213; // gcd(k_sig, p - 1) must be 1
    ll r = modExp(g, k_sig, p);
    ll k_inv = modInverse(k_sig, p - 1);
    ll s = (k_inv * (m_sig - x * r)) % (p - 1);
    if (s < 0) s += (p - 1);

    // Verification check: (h^r * r^s) mod p == g^m mod p
    ll v1 = (modExp(h, r, p) * modExp(r, s, p)) % p;
    ll v2 = modExp(g, m_sig, p);
    cout << "Signature Verified: " << (v1 == v2 ? "TRUE" : "FALSE") << "\n";

    return 0;
}