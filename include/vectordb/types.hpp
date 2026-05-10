#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

namespace vectordb {

using label_t = int64_t;
using id_t    = uint32_t;   // internal node id, supports up to 4B vectors

enum class Metric {
    L2,
    InnerProduct,
};

inline Metric parse_metric(const std::string& s) {
    if (s == "l2" || s == "L2") return Metric::L2;
    if (s == "ip" || s == "inner_product") return Metric::InnerProduct;
    throw std::invalid_argument("unknown metric: " + s);
}

}  // namespace vectordb
