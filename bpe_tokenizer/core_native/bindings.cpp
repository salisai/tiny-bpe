#include "bpe.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <thread>
#include <unordered_map>

namespace py = pybind11;
namespace b = bpe;

namespace {

// Zero-copy view into a py::bytes object; the owner keeps the ref alive.
struct Piece {
    const char* data;
    size_t len;
};

struct FastEncoder {
    explicit FastEncoder(b::Encoder encoder)
        : enc(std::move(encoder)) {}

    std::vector<uint32_t> encode_chunk(const std::vector<uint32_t>& ids) {
        py::gil_scoped_release release;
        return enc.encode_chunk(ids);
    }

    // One text = one list of UTF-8 byte pieces; returns the concatenated ids.
    std::vector<uint32_t> encode_pieces(const py::list& pieces) {
        std::vector<py::bytes> holders;
        holders.reserve(pieces.size());
        std::vector<Piece> pieces_c;
        pieces_c.reserve(pieces.size());
        for (py::handle p : pieces) {
            py::bytes b = p.cast<py::bytes>();
            char* ptr;
            Py_ssize_t n;
            if (PyBytes_AsStringAndSize(b.ptr(), &ptr, &n) != 0)
                throw py::error_already_set();
            pieces_c.push_back({ptr, size_t(n)});
            holders.push_back(std::move(b));
        }

        std::vector<uint32_t> result;
        {
            py::gil_scoped_release release;
            for (const Piece& p : pieces_c) {
                std::vector<uint32_t> ids;
                ids.reserve(p.len);
                for (size_t i = 0; i < p.len; ++i)
                    ids.push_back(uint8_t(p.data[i]));
                const std::vector<uint32_t> encoded = enc.encode_chunk(ids);
                result.insert(result.end(), encoded.begin(), encoded.end());
            }
        }
        return result;
    }

    // texts[i] = list of UTF-8 byte pieces for one text.
    // num_threads <= 0 means "pick automatically".
    std::vector<std::vector<uint32_t>> encode_batch(const py::list& texts,
                                                    int num_threads) {
        const size_t n_texts = size_t(texts.size());
        std::vector<std::vector<py::bytes>> holders(n_texts);
        std::vector<std::vector<Piece>> texts_c(n_texts);
        for (size_t t = 0; t < n_texts; ++t) {
            py::list text = texts[t].cast<py::list>();
            texts_c[t].reserve(text.size());
            holders[t].reserve(text.size());
            for (py::handle p : text) {
                py::bytes b = p.cast<py::bytes>();
                char* ptr;
                Py_ssize_t n;
                if (PyBytes_AsStringAndSize(b.ptr(), &ptr, &n) != 0)
                    throw py::error_already_set();
                texts_c[t].push_back({ptr, size_t(n)});
                holders[t].push_back(std::move(b));
            }
        }

        ensure_pool(num_threads);

        std::vector<std::vector<uint32_t>> results(n_texts);
        {
            py::gil_scoped_release release;
            if (n_texts >= kMinParallelTexts && pool_ && pool_->size() > 1) {
                pool_->parallel_for(n_texts, [&](size_t t) {
                    results[t] = encode_text(texts_c[t]);
                });
            } else {
                for (size_t t = 0; t < n_texts; ++t)
                    results[t] = encode_text(texts_c[t]);
            }
        }
        return results;
    }

private:
    std::vector<uint32_t> encode_text(const std::vector<Piece>& pieces) {
        std::vector<uint32_t> result;
        for (const Piece& p : pieces) {
            std::vector<uint32_t> ids;
            ids.reserve(p.len);
            for (size_t i = 0; i < p.len; ++i)
                ids.push_back(uint8_t(p.data[i]));
            const std::vector<uint32_t> encoded = enc.encode_chunk(ids);
            result.insert(result.end(), encoded.begin(), encoded.end());
        }
        return result;
    }

    void ensure_pool(int num_threads) {
        size_t want = size_t(num_threads);
        if (num_threads <= 0) {
            want = std::thread::hardware_concurrency();
            if (want == 0) want = 1;
        }
        if (!pool_ || pool_->size() != want) {
            pool_ = std::make_unique<b::ThreadPool>(want);
        }
    }

    static constexpr size_t kMinParallelTexts = 4;

    b::Encoder enc;
    std::unique_ptr<b::ThreadPool> pool_;
};

// Build a FastEncoder from a {(a, b): out_id} dict.
FastEncoder encoder_from_merges(const py::dict& merges) {
    std::vector<std::pair<uint64_t, uint64_t>> entries;
    entries.reserve(merges.size());
    for (auto item : merges) {
        const py::tuple pair = item.first.cast<py::tuple>();
        const uint32_t a = pair[0].cast<uint32_t>();
        const uint32_t b = pair[1].cast<uint32_t>();
        const uint32_t out = item.second.cast<uint32_t>();
        entries.emplace_back(b::key(a, b), b::rank_out(out - 256, out));
    }
    return FastEncoder(b::Encoder(std::move(entries)));
}

}  // namespace

PYBIND11_MODULE(core_native, m) {
    m.doc() =
        "C++ encoding core for the BPE tokenizer: GIL-released, thread-pooled "
        "greedy BPE encode.";

    py::class_<FastEncoder>(m, "FastEncoder",
                            "Prebuilt greedy BPE encoder over a merge table.")
        .def(py::init(&encoder_from_merges),
             py::arg("merges"),
             "merges: {(a, b): out_token_id}, ids assigned in merge order "
             "starting at 256.")
        .def("encode_chunk", &FastEncoder::encode_chunk, py::arg("ids"),
             "Encode one pre-tokenized piece (list of byte/token ids).")
        .def("encode_pieces", &FastEncoder::encode_pieces, py::arg("pieces"),
             "Encode a text given as a list of UTF-8 piece bytes; returns "
             "concatenated token ids. GIL released.")
        .def("encode_batch", &FastEncoder::encode_batch,
             py::arg("texts"), py::arg("num_threads") = 0,
             "Encode many texts (each a list of UTF-8 piece bytes) with a "
             "thread pool; GIL released. num_threads <= 0 picks "
             "hardware_concurrency.");

    m.def(
        "encode_chunk",
        [](const py::list& ids, const py::dict& ranks, const py::dict& merge_out) {
            // Faithful port of bpe_tokenizer/encode.py::encode_chunk:
            // rank from `ranks`, output id from `merge_out`.
            std::unordered_map<uint64_t, uint32_t> rank_lookup;
            rank_lookup.reserve(ranks.size());
            for (auto item : ranks) {
                const py::tuple pair = item.first.cast<py::tuple>();
                const uint64_t k = b::key(pair[0].cast<uint32_t>(),
                                          pair[1].cast<uint32_t>());
                rank_lookup[k] = item.second.cast<uint32_t>();
            }
            std::vector<std::pair<uint64_t, uint64_t>> entries;
            entries.reserve(merge_out.size());
            for (auto item : merge_out) {
                const py::tuple pair = item.first.cast<py::tuple>();
                const uint32_t a = pair[0].cast<uint32_t>();
                const uint32_t b = pair[1].cast<uint32_t>();
                const uint32_t out = item.second.cast<uint32_t>();
                const uint64_t k = b::key(a, b);
                auto it = rank_lookup.find(k);
                const uint32_t rank =
                    it != rank_lookup.end() ? it->second : out - 256;
                entries.emplace_back(k, b::rank_out(rank, out));
            }
            const b::Encoder enc(std::move(entries));

            std::vector<uint32_t> ids_c;
            ids_c.reserve(ids.size());
            for (py::handle i : ids) ids_c.push_back(i.cast<uint32_t>());

            std::vector<uint32_t> result;
            {
                py::gil_scoped_release release;
                result = enc.encode_chunk(ids_c);
            }
            return result;
        },
        py::arg("ids"), py::arg("ranks"), py::arg("merge_out"),
        "Port of encode.py::encode_chunk: greedy BPE on one piece.");
}
