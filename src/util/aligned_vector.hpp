#ifndef ALIGNED_VECTOR_HPP
#define ALIGNED_VECTOR_HPP

#include <cstddef>
#include <limits>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

template <class T, std::size_t Alignment = 64>
struct AlignedAllocator {
    using value_type = T;
    using is_always_equal = std::true_type;

    AlignedAllocator() noexcept = default;
    template <class U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    [[nodiscard]] T* allocate(std::size_t n) {
        static_assert((Alignment & (Alignment - 1)) == 0, "Alignment must be power of two");
        static_assert(Alignment >= alignof(T) && (Alignment % alignof(T) == 0),
                      "Alignment must be a multiple of alignof(T)");

        if (n == 0) return nullptr;
        if (n > (std::numeric_limits<std::size_t>::max)() / sizeof(T)) throw std::bad_alloc();

        void* p = ::operator new(n * sizeof(T), std::align_val_t{Alignment});
        return static_cast<T*>(p);
    }

    template <class U>
    void construct(U* p) noexcept(std::is_nothrow_default_constructible_v<U>) {
        ::new (static_cast<void*>(p)) U;
    }

    template <class U, class Arg0, class... Args>
    void construct(U* p, Arg0&& arg0, Args&&... args)
        noexcept(std::is_nothrow_constructible_v<U, Arg0&&, Args&&...>) {
        ::new (static_cast<void*>(p)) U(std::forward<Arg0>(arg0), std::forward<Args>(args)...);
    }

    void deallocate(T* p, std::size_t) noexcept {
        ::operator delete(p, std::align_val_t{Alignment});
    }

    template <class U>
    struct rebind { using other = AlignedAllocator<U, Alignment>; };

    bool operator==(const AlignedAllocator&) const noexcept { return true; }
    bool operator!=(const AlignedAllocator&) const noexcept { return false; }
};

template <class T, std::size_t Alignment = 64>
using MyVector = std::vector<T, AlignedAllocator<T, Alignment>>;

#endif
