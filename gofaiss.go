//go:build darwin

package gofaiss

/*
#cgo CXXFLAGS: -std=c++17
#cgo CFLAGS: -I./faisscapi
#cgo LDFLAGS: -L./faisslib/v1.11.0/darwin -lfaiss_c -lfaiss -Wl,-rpath,./faisscapi

#include <stdlib.h>
#include <faiss_c.h>
#include <error_c.h>
#include <Index_c.h>
#include <IndexFlat_c.h>
#include <index_factory_c.h>
#include <index_io_c.h>
#include <IndexIVF_c.h>
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// MetricType defines the metric used for similarity search.
type MetricType int

const (
	// MetricInnerProduct is for maximum inner product search.
	MetricInnerProduct MetricType = 0
	// MetricL2 is for squared L2 search.
	MetricL2 MetricType = 1
	// MetricL1 is for L1 (cityblock) distance.
	MetricL1 MetricType = 2
	// MetricLinf is for infinity distance.
	MetricLinf MetricType = 3
	// MetricLp is for Lp distance.
	MetricLp MetricType = 4

	// MetricCanberra is for Canberra distance.
	MetricCanberra MetricType = 20
	// MetricBrayCurtis is for Bray-Curtis distance.
	MetricBrayCurtis MetricType = 21
	// MetricJensenShannon is for Jensen-Shannon divergence.
	MetricJensenShannon MetricType = 22
)

// Index represents a Faiss index.
type Index struct {
	c_index *C.FaissIndex
}

// NewIndexFlatL2 creates a new IndexFlatL2.
func NewIndexFlatL2(d int) (*Index, error) {
	var c_index *C.FaissIndex
	ret := C.faiss_IndexFlatL2_new_with(&c_index, C.longlong(d))
	if ret != 0 {
		return nil, getLastError()
	}

	return &Index{c_index: c_index}, nil
}

// NewIndex creates an index from a factory string.
func NewIndex(d int, description string, metric MetricType) (*Index, error) {
	var c_index *C.FaissIndex
	c_description := C.CString(description)
	defer C.free(unsafe.Pointer(c_description))

	ret := C.faiss_index_factory(&c_index, C.int(d), c_description, C.FaissMetricType(metric))
	if ret != 0 {
		return nil, getLastError()
	}

	return &Index{c_index: c_index}, nil
}

// ReadIndex reads an index from a file.
func ReadIndex(filename string) (*Index, error) {
	var c_index *C.FaissIndex
	c_filename := C.CString(filename)
	defer C.free(unsafe.Pointer(c_filename))

	// io_flags, 0 for default
	ret := C.faiss_read_index_fname(c_filename, 0, &c_index)
	if ret != 0 {
		return nil, getLastError()
	}

	return &Index{c_index: c_index}, nil
}

// Write writes an index to a file.
func (idx *Index) Write(filename string) error {
	c_filename := C.CString(filename)
	defer C.free(unsafe.Pointer(c_filename))

	ret := C.faiss_write_index_fname(idx.c_index, c_filename)
	if ret != 0 {
		return getLastError()
	}
	return nil
}

// D returns the dimension of the index.
func (idx *Index) D() int {
	return int(C.faiss_Index_d(idx.c_index))
}

// IsTrained returns whether the index is trained.
func (idx *Index) IsTrained() bool {
	return C.faiss_Index_is_trained(idx.c_index) != 0
}

// Ntotal returns the number of vectors in the index.
func (idx *Index) Ntotal() int64 {
	return int64(C.faiss_Index_ntotal(idx.c_index))
}

// Train trains the index on a set of vectors.
func (idx *Index) Train(x []float32) error {
	if len(x) == 0 {
		return nil
	}
	n := len(x) / idx.D()
	ret := C.faiss_Index_train(idx.c_index, C.longlong(n), (*C.float)(unsafe.Pointer(&x[0])))
	if ret != 0 {
		return getLastError()
	}
	return nil
}

// Add adds vectors to the index.
func (idx *Index) Add(x []float32) error {
	if len(x) == 0 {
		return nil
	}
	n := len(x) / idx.D()
	ret := C.faiss_Index_add(idx.c_index, C.longlong(n), (*C.float)(unsafe.Pointer(&x[0])))
	if ret != 0 {
		return getLastError()
	}
	return nil
}

// Search searches for the k nearest neighbors of the given vectors.
func (idx *Index) Search(x []float32, k int) (distances []float32, labels []int64, err error) {
	n := len(x) / idx.D()
	if n == 0 {
		return nil, nil, nil
	}

	distances = make([]float32, n*k)
	labels = make([]int64, n*k)

	ret := C.faiss_Index_search(
		idx.c_index,
		C.longlong(n),
		(*C.float)(unsafe.Pointer(&x[0])),
		C.longlong(k),
		(*C.float)(unsafe.Pointer(&distances[0])),
		(*C.longlong)(unsafe.Pointer(&labels[0])),
	)

	if ret != 0 {
		return nil, nil, getLastError()
	}

	return distances, labels, nil
}

// Free frees the memory associated with the index.
func (idx *Index) Free() {
	if idx.c_index != nil {
		C.faiss_Index_free(idx.c_index)
		idx.c_index = nil
	}
}

func getLastError() error {
	err_msg := C.faiss_get_last_error()
	if err_msg == nil {
		return nil
	}
	err := C.GoString(err_msg)
	return fmt.Errorf("faiss: %s", err)
}
