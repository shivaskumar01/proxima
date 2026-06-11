#include "vectordb/flat.hpp"
#include "vectordb/hnsw.hpp"
#include "vectordb/ivfpq.hpp"
#include "vectordb/types.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <utility>

namespace py = pybind11;
using namespace vectordb;

namespace {

// Force the input array into row-major contiguous float32. forcecast handles
// dtype conversion; c_style guarantees stride is dim*sizeof(float) so we can
// pass the raw pointer to the C++ kernels.
using FloatArr = py::array_t<float, py::array::c_style | py::array::forcecast>;

const float* check_2d_dim(const FloatArr& arr, std::size_t dim,
                          const char* name, std::size_t* n_out) {
    auto buf = arr.request();
    if (buf.ndim != 2 || buf.shape[1] != static_cast<py::ssize_t>(dim)) {
        throw std::invalid_argument(std::string(name) + " must be 2-D with shape (n, dim)");
    }
    *n_out = static_cast<std::size_t>(buf.shape[0]);
    return static_cast<const float*>(buf.ptr);
}

template <typename Idx>
auto py_search(const Idx& self, FloatArr queries, std::size_t k) {
    std::size_t nq;
    const float* q = check_2d_dim(queries, self.dim(), "queries", &nq);
    py::array_t<float>   dists({nq, k});
    py::array_t<label_t> labels({nq, k});
    // Resolve buffer pointers BEFORE releasing the GIL — request() calls back
    // into Python and segfaults if the GIL is dropped.
    float*   d_ptr = static_cast<float*>(dists.request().ptr);
    label_t* l_ptr = static_cast<label_t*>(labels.request().ptr);
    {
        py::gil_scoped_release release;
        self.search(q, nq, k, d_ptr, l_ptr);
    }
    return std::make_pair(dists, labels);
}

}  // namespace

PYBIND11_MODULE(_vectordb, m) {
    m.doc() = "vectordb: HNSW and IVF-PQ indexes (C++ core)";

    // ---- FlatIndex --------------------------------------------------------
    py::class_<FlatIndex>(m, "FlatIndex")
        .def(py::init([](std::size_t dim, std::string metric) {
                 return new FlatIndex(dim, parse_metric(metric));
             }),
             py::arg("dim"), py::arg("metric") = "l2")
        .def("add",
             [](FlatIndex& self, FloatArr data) {
                 std::size_t n;
                 const float* p = check_2d_dim(data, self.dim(), "data", &n);
                 py::gil_scoped_release release;
                 self.add(p, n);
             },
             py::arg("data"))
        .def("search", &py_search<FlatIndex>,
             py::arg("queries"), py::arg("k"))
        .def("reserve", &FlatIndex::reserve, py::arg("n"),
             "Pre-size storage for n total vectors (avoids 2x realloc peaks "
             "when streaming chunked add() calls).")
        .def("save", &FlatIndex::save, py::arg("path"))
        .def_static("load", &FlatIndex::load, py::arg("path"))
        .def_property_readonly("size", &FlatIndex::size)
        .def_property_readonly("dim",  &FlatIndex::dim);

    // ---- HnswIndex --------------------------------------------------------
    py::class_<HnswIndex>(m, "HnswIndex")
        .def(py::init([](std::size_t dim, std::string metric,
                          std::size_t M, std::size_t ef_construction,
                          uint64_t seed) {
                 return new HnswIndex(dim, parse_metric(metric),
                                      M, ef_construction, seed);
             }),
             py::arg("dim"),
             py::arg("metric") = "l2",
             py::arg("M") = 16,
             py::arg("ef_construction") = 200,
             py::arg("seed") = 42)
        .def("add",
             [](HnswIndex& self, FloatArr data) {
                 std::size_t n;
                 const float* p = check_2d_dim(data, self.dim(), "data", &n);
                 py::gil_scoped_release release;
                 self.add(p, n);
             },
             py::arg("data"))
        .def("search",
             [](const HnswIndex& self, FloatArr queries,
                std::size_t k, std::size_t ef) {
                 std::size_t nq;
                 const float* q = check_2d_dim(queries, self.dim(), "queries", &nq);
                 py::array_t<float>   dists({nq, k});
                 py::array_t<label_t> labels({nq, k});
                 float*   d_ptr = static_cast<float*>(dists.request().ptr);
                 label_t* l_ptr = static_cast<label_t*>(labels.request().ptr);
                 {
                     py::gil_scoped_release release;
                     self.search(q, nq, k, ef, d_ptr, l_ptr);
                 }
                 return std::make_pair(dists, labels);
             },
             py::arg("queries"), py::arg("k"), py::arg("ef") = 64)
        .def("reserve", &HnswIndex::reserve, py::arg("n"),
             "Pre-size storage for n total vectors (avoids 2x realloc peaks "
             "when streaming chunked add() calls).")
        .def("save", &HnswIndex::save, py::arg("path"))
        .def_static("load", &HnswIndex::load, py::arg("path"))
        .def_property_readonly("size", &HnswIndex::size)
        .def_property_readonly("dim",  &HnswIndex::dim)
        .def("level_histogram", &HnswIndex::level_histogram);

    // ---- IvfPqIndex -------------------------------------------------------
    py::class_<IvfPqIndex>(m, "IvfPqIndex")
        .def(py::init<std::size_t, std::size_t, std::size_t,
                      std::size_t, uint64_t>(),
             py::arg("dim"), py::arg("nlist"), py::arg("M"),
             py::arg("kmeans_iters") = 20, py::arg("seed") = 42)
        .def("train",
             [](IvfPqIndex& self, FloatArr data) {
                 std::size_t n;
                 const float* p = check_2d_dim(data, self.dim(), "data", &n);
                 py::gil_scoped_release release;
                 self.train(p, n);
             },
             py::arg("data"))
        .def("add",
             [](IvfPqIndex& self, FloatArr data) {
                 std::size_t n;
                 const float* p = check_2d_dim(data, self.dim(), "data", &n);
                 py::gil_scoped_release release;
                 self.add(p, n);
             },
             py::arg("data"))
        .def("search",
             [](const IvfPqIndex& self, FloatArr queries,
                std::size_t k, std::size_t nprobe) {
                 std::size_t nq;
                 const float* q = check_2d_dim(queries, self.dim(), "queries", &nq);
                 py::array_t<float>   dists({nq, k});
                 py::array_t<label_t> labels({nq, k});
                 float*   d_ptr = static_cast<float*>(dists.request().ptr);
                 label_t* l_ptr = static_cast<label_t*>(labels.request().ptr);
                 {
                     py::gil_scoped_release release;
                     self.search(q, nq, k, nprobe, d_ptr, l_ptr);
                 }
                 return std::make_pair(dists, labels);
             },
             py::arg("queries"), py::arg("k"), py::arg("nprobe") = 8)
        .def("save", &IvfPqIndex::save, py::arg("path"))
        .def_static("load", &IvfPqIndex::load, py::arg("path"))
        .def_property_readonly("is_trained", &IvfPqIndex::is_trained)
        .def_property_readonly("size",  &IvfPqIndex::size)
        .def_property_readonly("dim",   &IvfPqIndex::dim)
        .def_property_readonly("nlist", &IvfPqIndex::nlist)
        .def_property_readonly("M",     &IvfPqIndex::M)
        .def("list_sizes", &IvfPqIndex::list_sizes);
}
