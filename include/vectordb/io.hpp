#pragma once

// Tiny binary I/O helpers for index serialization.
//
// All numeric fields are little-endian POD writes — Apple Silicon and x86-64
// are both LE so we don't byte-swap. Files are NOT cross-version: the magic+
// version header detects mismatches and throws.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

namespace vectordb {
namespace io {

class Writer {
public:
    explicit Writer(const std::string& path) {
        f_ = std::fopen(path.c_str(), "wb");
        if (!f_) throw std::runtime_error("vectordb::io: cannot open for writing: " + path);
    }
    ~Writer() { if (f_) std::fclose(f_); }
    Writer(const Writer&) = delete;
    Writer& operator=(const Writer&) = delete;

    void write_raw(const void* p, std::size_t n) {
        if (n == 0) return;
        if (std::fwrite(p, 1, n, f_) != n) {
            throw std::runtime_error("vectordb::io: write failed");
        }
    }
    template <typename T>
    void write_pod(T v) { write_raw(&v, sizeof(T)); }

    void write_magic(const char m[4]) { write_raw(m, 4); }

private:
    std::FILE* f_ = nullptr;
};

class Reader {
public:
    explicit Reader(const std::string& path) {
        f_ = std::fopen(path.c_str(), "rb");
        if (!f_) throw std::runtime_error("vectordb::io: cannot open for reading: " + path);
    }
    ~Reader() { if (f_) std::fclose(f_); }
    Reader(const Reader&) = delete;
    Reader& operator=(const Reader&) = delete;

    void read_raw(void* p, std::size_t n) {
        if (n == 0) return;
        if (std::fread(p, 1, n, f_) != n) {
            throw std::runtime_error("vectordb::io: read failed (truncated file?)");
        }
    }
    template <typename T>
    T read_pod() { T v; read_raw(&v, sizeof(T)); return v; }

    void check_magic(const char expected[4]) {
        char m[4];
        read_raw(m, 4);
        if (std::memcmp(m, expected, 4) != 0) {
            throw std::runtime_error("vectordb::io: bad magic header");
        }
    }

private:
    std::FILE* f_ = nullptr;
};

}  // namespace io
}  // namespace vectordb
