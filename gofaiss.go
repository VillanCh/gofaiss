//go:build darwin

package gofaiss

/*
#cgo CFLAGS: -I./faisscapi
#cgo LDFLAGS: -L./faisslib/v1.11.0/darwin -lfaiss_c -lfaiss -Wl,-rpath,./faisslib/v1.11.0/darwin
#include <IndexFlat_c.h>
#include <Index_c.h>
#include <error_c.h>
#include <stdlib.h>
*/
import "C"

import (
	"fmt"
	"unsafe"
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
