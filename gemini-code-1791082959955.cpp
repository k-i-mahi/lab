#include <iostream>
#include <vector>
#include <string>

using namespace std;

// 1. Caesar Cipher
string caesarEnc(string s, int k) {
    for (char &c : s) if (isalpha(c)) {
        char base = isupper(c) ? 'A' : 'a';
        c = (c - base + k) % 26 + base;
    }
    return s;
}
string caesarDec(string s, int k) { return caesarEnc(s, 26 - (k % 26)); }

// 2. Vernam Cipher (One-Time Pad / XOR)
string vernam(string text, string key) {
    string res = text;
    for (size_t i = 0; i < text.size(); ++i) res[i] ^= key[i % key.size()];
    return res;
}

// 3. Matrix Transposition
string matrixTransposeEnc(string s, int cols) {
    int rows = (s.size() + cols - 1) / cols;
    s.resize(rows * cols, 'X'); // padding
    string res = "";
    for (int c = 0; c < cols; ++c)
        for (int r = 0; r < rows; ++r)
            res += s[r * cols + c];
    return res;
}

string matrixTransposeDec(string s, int cols) {
    int rows = s.size() / cols;
    string res(s.size(), ' ');
    int idx = 0;
    for (int c = 0; c < cols; ++c)
        for (int r = 0; r < rows; ++r)
            res[r * cols + c] = s[idx++];
    return res;
}

int main() {
    string msg = "SECURITY_LAB", key = "SECRETKEY123";

    // Caesar Test
    string c_enc = caesarEnc(msg, 3);
    cout << "[Caesar] Enc: " << c_enc << " | Dec: " << caesarDec(c_enc, 3) << "\n";

    // Vernam Test
    string v_enc = vernam(msg, key);
    cout << "[Vernam] Enc (raw bytes) | Dec: " << vernam(v_enc, key) << "\n";

    // Vernam + Matrix Transposition Test
    string v_trans_enc = matrixTransposeEnc(vernam(msg, key), 4);
    string v_trans_dec = vernam(matrixTransposeDec(v_trans_enc, 4), key);
    while (!v_trans_dec.empty() && v_trans_dec.back() == 'X') v_trans_dec.pop_back();
    cout << "[Vernam + Transposition] Dec: " << v_trans_dec << "\n";

    return 0;
}