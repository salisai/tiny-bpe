#include "bpe.hpp"

#include <condition_variable>
#include <queue>
#include <thread>

namespace bpe {

FlatTable::FlatTable(size_t expected) {
    size_t cap = 16;
    while (cap < expected * 2) cap <<= 1;
    keys_.assign(cap, EMPTY);
    vals_.assign(cap, 0);
}

void FlatTable::insert(uint64_t k, uint64_t v) {
    size_t idx = hash(k) & (keys_.size() - 1);
    while (keys_[idx] != EMPTY) {
        if (keys_[idx] == k) {
            vals_[idx] = v;
            return;
        }
        idx = (idx + 1) & (keys_.size() - 1);
    }
    keys_[idx] = k;
    vals_[idx] = v;
}

bool FlatTable::find(uint64_t k, uint64_t* out) const {
    size_t idx = hash(k) & (keys_.size() - 1);
    while (keys_[idx] != EMPTY) {
        if (keys_[idx] == k) {
            *out = vals_[idx];
            return true;
        }
        idx = (idx + 1) & (keys_.size() - 1);
    }
    return false;
}

size_t FlatTable::hash(uint64_t k) {
    k ^= k >> 30;
    k *= 0xbf58476d1ce4e5b9ull;
    k ^= k >> 27;
    k *= 0x94d049bb133111ebull;
    k ^= k >> 31;
    return size_t(k);
}

Encoder::Encoder(std::vector<std::pair<uint64_t, uint64_t>> entries)
    : table_(entries.size()) {
    for (const auto& e : entries) table_.insert(e.first, e.second);
}

std::vector<uint32_t> Encoder::encode_chunk(
    const std::vector<uint32_t>& ids) const {
    const size_t n = ids.size();
    if (n < 2) return ids;

    std::vector<uint32_t> val = ids;
    std::vector<int32_t> nxt(n), prev(n);
    for (size_t i = 0; i + 1 < n; ++i) nxt[i] = int32_t(i + 1);
    nxt[n - 1] = -1;
    prev[0] = -1;
    for (size_t i = 1; i < n; ++i) prev[i] = int32_t(i - 1);
    std::vector<uint8_t> alive(n, 1);

    using Entry = std::pair<int32_t, int32_t>;  // (rank, position)
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> heap;

    const auto try_push = [&](int32_t i) {
        if (i < 0 || !alive[i]) return;
        const int32_t j = nxt[i];
        if (j < 0 || !alive[j]) return;
        uint64_t v;
        if (table_.find(key(val[i], val[j]), &v)) {
            heap.push({int32_t(rank_of(v)), i});
        }
    };

    for (size_t i = 0; i + 1 < n; ++i) try_push(int32_t(i));

    while (!heap.empty()) {
        const Entry e = heap.top();
        heap.pop();
        const int32_t i = e.second;
        if (!alive[i]) continue;
        const int32_t j = nxt[i];
        if (j < 0 || !alive[j]) continue;
        uint64_t v;
        if (!table_.find(key(val[i], val[j]), &v)) continue;
        if (int32_t(rank_of(v)) != e.first) continue;  // stale heap entry
        val[i] = out_of(v);
        const int32_t k = nxt[j];
        nxt[i] = k;
        if (k >= 0) prev[k] = i;
        alive[j] = 0;
        try_push(prev[i]);
        try_push(i);
    }

    std::vector<uint32_t> result;
    result.reserve(val.size());
    int32_t i = 0;
    while (i >= 0) {
        result.push_back(val[i]);
        i = nxt[i];
    }
    return result;
}

ThreadPool::ThreadPool(size_t n) : stop_(false) {
    workers_.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        workers_.emplace_back([this] { loop(); });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lk(m_);
        stop_ = true;
    }
    cv_.notify_all();
    for (auto& w : workers_) w.join();
}

void ThreadPool::submit(std::function<void()> f) {
    {
        std::unique_lock<std::mutex> lk(m_);
        tasks_.push(std::move(f));
    }
    cv_.notify_one();
}

void ThreadPool::loop() {
    for (;;) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lk(m_);
            cv_.wait(lk, [this] { return stop_ || !tasks_.empty(); });
            if (stop_ && tasks_.empty()) return;
            task = std::move(tasks_.front());
            tasks_.pop();
        }
        task();
    }
}

void ThreadPool::parallel_for(size_t n,
                              const std::function<void(size_t)>& f) {
    const size_t nw = std::min(n, workers_.size());
    if (nw == 0) {
        for (size_t i = 0; i < n; ++i) f(i);
        return;
    }
    std::atomic<size_t> next{0};
    std::mutex done_m;
    std::condition_variable done_cv;
    size_t remaining = nw;  // one decrement per submitted task
    for (size_t w = 0; w < nw; ++w) {
        submit([&, w] {
            for (;;) {
                const size_t i = next.fetch_add(1);
                if (i >= n) break;
                f(i);
            }
            std::unique_lock<std::mutex> lk(done_m);
            if (--remaining == 0) done_cv.notify_one();
        });
    }
    std::unique_lock<std::mutex> lk(done_m);
    done_cv.wait(lk, [&] { return remaining == 0; });
}

}  // namespace bpe
