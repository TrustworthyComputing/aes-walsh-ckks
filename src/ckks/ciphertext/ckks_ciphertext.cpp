#include "ckks_ciphertext.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>




CKKSCiphertext::CKKSCiphertext(size_t N,
                               MyVector<uint64_t> a,
                               MyVector<uint64_t> b,
                               long double sc,
                               size_t lv,
                               CKKSSecretOwner owner,
                               CKKSMessageEncodingState encoding_state){
    this->N = N;
    polys.reserve(2);
    // Store as [b, a] so poly[0] is the constant term.
    polys.emplace_back(std::move(b));
    polys.emplace_back(std::move(a));
    this->scale = sc;
    this->level = lv;
    this->secret_owner = owner;
    this->message_encoding_state = encoding_state;
}

CKKSCiphertext::CKKSCiphertext(size_t N,
                               MyVector<MyVector<uint64_t>> polys_in,
                               long double sc,
                               size_t lv,
                               CKKSSecretOwner owner,
                               CKKSMessageEncodingState encoding_state){
    this->N = N;
    polys = std::move(polys_in);
    this->scale = sc;
    this->level = lv;
    this->secret_owner = owner;
    this->message_encoding_state = encoding_state;
}

CKKSCiphertext CKKSCiphertext::clone() const {
    return CKKSCiphertext(N, polys, scale, level, secret_owner, message_encoding_state);
}

long double CKKSCiphertext::getScale() const {
    return scale;
}

size_t CKKSCiphertext::getLevel() const {
    return level;
}

size_t CKKSCiphertext::getN() const {
    return N;
}

size_t CKKSCiphertext::getNumPolys() const {
    return polys.size();
}

CKKSSecretOwner CKKSCiphertext::getSecretOwner() const {
    return secret_owner;
}

CKKSMessageEncodingState CKKSCiphertext::getMessageEncodingState() const {
    return message_encoding_state;
}

const MyVector<uint64_t>& CKKSCiphertext::getA() const {
    if (polys.size() < 2) {
        throw std::runtime_error("CKKSCiphertext: missing A polynomial");
    }
    return polys[1];
}

const MyVector<uint64_t>& CKKSCiphertext::getB() const {
    if (polys.size() < 1) {
        throw std::runtime_error("CKKSCiphertext: missing B polynomial");
    }
    return polys[0];
}

MyVector<uint64_t>& CKKSCiphertext::getAMutable() {
    if (polys.size() < 2) {
        throw std::runtime_error("CKKSCiphertext: missing A polynomial");
    }
    return polys[1];
}

MyVector<uint64_t>& CKKSCiphertext::getBMutable() {
    if (polys.size() < 1) {
        throw std::runtime_error("CKKSCiphertext: missing B polynomial");
    }
    return polys[0];
}

const MyVector<uint64_t>& CKKSCiphertext::getPoly(size_t idx) const {
    return polys.at(idx);
}

MyVector<uint64_t>& CKKSCiphertext::getPolyMutable(size_t idx) {
    return polys.at(idx);
}

void CKKSCiphertext::addPoly(MyVector<uint64_t> poly) {
    if (polys.capacity() < polys.size() + 1) {
        polys.reserve(polys.size() + 1);
    }
    polys.emplace_back(std::move(poly));
}

void CKKSCiphertext::setScale(long double new_scale) {
    scale = new_scale;
}

void CKKSCiphertext::setLevel(size_t new_level) {
    level = new_level;
}

void CKKSCiphertext::setSecretOwner(CKKSSecretOwner new_secret_owner) {
    secret_owner = new_secret_owner;
}

void CKKSCiphertext::setMessageEncodingState(CKKSMessageEncodingState new_message_encoding_state) {
    message_encoding_state = new_message_encoding_state;
}

MyVector<MyVector<uint64_t>>& CKKSCiphertext::getPolysMutable() {
    return polys;
}

// CKKS ENCODING _______________________________________________________
// ######################################################################

CKKSEncoding::CKKSEncoding(
    size_t N,
    MyVector<uint64_t> plaintext,
    long double sc,
    size_t lv,
    bool montgomery){
    this->N = N;
    this->plaintext = std::move(plaintext);
    this->scale = sc;
    this->level = lv;
    this->montgomery_form = montgomery;
}

const MyVector<uint64_t>& CKKSEncoding::getPlaintext() const{
    return plaintext;
}

bool CKKSEncoding::isMontgomeryForm() const {
    return montgomery_form;
}

long double CKKSEncoding::getScale() const {
    return scale;
}

size_t CKKSEncoding::getLevel() const {
    return level;
}

size_t CKKSEncoding::getN() const {
    return N;
}
