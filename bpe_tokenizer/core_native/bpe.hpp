#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

namespace bpe {

// Pack an adjacent token pair (a, b) into one 64-bit key.
inline uint64_t key(uint32_t a, uint32_t b) {
    return (uint64_t(a) << 32) | uint32_t(b);
}

// Pack (rank, out_id) into one 64-bit value.
inline uint64_t rank_out(uint32_t rank, uint32_t out) {
    return (uint64_t(rank) << 32) | out;
}

inline uint32_t rank_of(uint64_t v) { return uint32_t(v >> 32); }
inline uint32_t out_of(uint64_t v) { return uint32_t(v); }

// Minimal open-addressing hash table: uint64 key -> uint64 value.
// Linear probing, power-of-two capacity, load factor capped at 0.5.
class FlatTable {
public:
    explicit FlatTable(size_t expected);

    void insert(uint64_t k, uint64_t v);
    bool find(uint64_t k, uint64_t* out) const;

private:
    static size_t hash(uint64_t k);

    static constexpr uint64_t EMPTY = ~0ull;
    std::vector<uint64_t> keys_;
    std::vector<uint64_t> vals_;
};

// Greedy byte-level BPE encoder: 1:1 port of bpe_tokenizer/encode.py
// (min-heap of (rank, position) + doubly-linked list of live nodes).
class Encoder {
public:
    // entries: (packed pair key, packed (rank, out_id)).
    explicit Encoder(std::vector<std::pair<uint64_t, uint64_t>> entries);

    // Encode one pre-tokenized piece of token ids.
    std::vector<uint32_t> encode_chunk(const std::vector<uint32_t>& ids) const;

private:
    FlatTable table_;
};

// Small persistent thread pool for batch encoding.
class ThreadPool {
public:
    explicit ThreadPool(size_t n);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    size_t size() const { return workers_.size(); }

    // Run f(i) for i in [0, n). Returns when all iterations are done.
    void parallel_for(size_t n, const std::function<void(size_t)>& f);

private:
    void loop();
    void submit(std::function<void()> f);

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex m_;
    std::condition_variable cv_;
    bool stop_;
};

}  // namespace bpe
