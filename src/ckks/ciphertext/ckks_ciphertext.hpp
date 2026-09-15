#ifndef CKKS_CIPHERTEXT_HPP
#define CKKS_CIPHERTEXT_HPP

#include <cstddef>
#include <cstdint>
#include "aligned_vector.hpp"

enum class CKKSSecretOwner {
    Dense,
    BootstrapSparse,
};

enum class CKKSMessageEncodingState {
    Slots,
    Coefficients,
};

class CKKSCiphertext{
public:
    CKKSCiphertext(size_t N,
                   MyVector<uint64_t> a,
                   MyVector<uint64_t> b,
                   long double scale,
                   size_t level,
                   CKKSSecretOwner secret_owner = CKKSSecretOwner::Dense,
                   CKKSMessageEncodingState message_encoding_state = CKKSMessageEncodingState::Slots);
    CKKSCiphertext(size_t N,
                   MyVector<MyVector<uint64_t>> polys,
                   long double scale,
                   size_t level,
                   CKKSSecretOwner secret_owner = CKKSSecretOwner::Dense,
                   CKKSMessageEncodingState message_encoding_state = CKKSMessageEncodingState::Slots);

    // disables copy constructor and copy assignment
    CKKSCiphertext(const CKKSCiphertext&) = delete;
    CKKSCiphertext& operator=(const CKKSCiphertext&) = delete;

    // enables move constructor and move assignment
    CKKSCiphertext(CKKSCiphertext&&) noexcept = default;
    CKKSCiphertext& operator=(CKKSCiphertext&&) noexcept = default;

    CKKSCiphertext clone() const;


    long double getScale() const;
    size_t getLevel() const;
    size_t getN() const;
    size_t getNumPolys() const;
    CKKSSecretOwner getSecretOwner() const;
    CKKSMessageEncodingState getMessageEncodingState() const;

    const MyVector<uint64_t>& getA() const;
    const MyVector<uint64_t>& getB() const;
    MyVector<uint64_t>& getAMutable();
    MyVector<uint64_t>& getBMutable();
    const MyVector<uint64_t>& getPoly(size_t idx) const;
    MyVector<uint64_t>& getPolyMutable(size_t idx);
    MyVector<MyVector<uint64_t>>& getPolysMutable();
    void addPoly(MyVector<uint64_t> poly);
    void setScale(long double new_scale);
    void setLevel(size_t new_level);
    void setSecretOwner(CKKSSecretOwner new_secret_owner);
    void setMessageEncodingState(CKKSMessageEncodingState new_message_encoding_state);


private:
    size_t N;
    long double scale;
    size_t level;
    CKKSSecretOwner secret_owner = CKKSSecretOwner::Dense;
    CKKSMessageEncodingState message_encoding_state = CKKSMessageEncodingState::Slots;
    MyVector<MyVector<uint64_t>> polys;
};


class CKKSEncoding {
public:
    CKKSEncoding(
        size_t N,
        MyVector<uint64_t> plaintext,
        long double scale,
        size_t level,
        bool montgomery_form = false);

    // disables copy constructor and copy assignment
    CKKSEncoding(const CKKSEncoding&) = delete;
    CKKSEncoding& operator=(const CKKSEncoding&) = delete;

    // enables move constructor and move assignment
    CKKSEncoding(CKKSEncoding&&) noexcept = default;
    CKKSEncoding& operator=(CKKSEncoding&&) noexcept = default;

    ~CKKSEncoding() = default;


    const MyVector<uint64_t>& getPlaintext() const;
    bool isMontgomeryForm() const;
    long double getScale() const;
    size_t getLevel() const;
    size_t getN() const;

private:
    size_t N;
    long double scale;
    size_t level;
    MyVector<uint64_t> plaintext;
    bool montgomery_form = false;
};

#endif  // CKKS_CIPHERTEXT_HPP
