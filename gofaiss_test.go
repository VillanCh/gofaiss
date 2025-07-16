package gofaiss

import (
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
)

func TestIndexFlatL2(t *testing.T) {
	d := 64    // dimension
	nb := 1000 // database size
	nq := 100  // number of queries
	k := 4     // number of nearest neighbors

	// generate database vectors
	xb := make([]float32, d*nb)
	for i := 0; i < nb*d; i++ {
		xb[i] = rand.Float32()
	}

	// generate query vectors
	xq := make([]float32, d*nq)
	for i := 0; i < nq*d; i++ {
		xq[i] = rand.Float32()
	}

	index, err := NewIndexFlatL2(d)
	if err != nil {
		t.Fatalf("NewIndexFlatL2 failed: %v", err)
	}
	defer index.Free()

	if index.D() != d {
		t.Fatalf("expected dimension %d, got %d", d, index.D())
	}
	if !index.IsTrained() {
		t.Fatal("expected index to be trained")
	}
	if index.Ntotal() != 0 {
		t.Fatalf("expected 0 vectors in index, got %d", index.Ntotal())
	}

	err = index.Add(xb)
	if err != nil {
		t.Fatalf("Add failed: %v", err)
	}
	if index.Ntotal() != int64(nb) {
		t.Fatalf("expected %d vectors in index, got %d", nb, index.Ntotal())
	}

	distances, labels, err := index.Search(xq, k)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(distances) != nq*k {
		t.Fatalf("expected %d distances, got %d", nq*k, len(distances))
	}
	if len(labels) != nq*k {
		t.Fatalf("expected %d labels, got %d", nq*k, len(labels))
	}

	t.Logf("Search successful. First label: %d, first distance: %f", labels[0], distances[0])
}

func TestIndexIVFFlat_FullLifecycle(t *testing.T) {
	d := 32      // dimension
	nb := 2000   // database size
	nq := 200    // number of queries
	nlist := 100 // number of IVF clusters
	k := 5       // number of nearest neighbors

	// generate database vectors
	xb := make([]float32, d*nb)
	for i := 0; i < nb*d; i++ {
		xb[i] = rand.Float32()
	}

	// generate query vectors
	xq := make([]float32, d*nq)
	for i := 0; i < nq*d; i++ {
		xq[i] = rand.Float32()
	}

	// 1. Create index
	index, err := NewIndex(d, fmt.Sprintf("IVF%d,Flat", nlist), MetricL2)
	if err != nil {
		t.Fatalf("NewIndex failed: %v", err)
	}
	defer index.Free()

	// 2. Train index
	// training on the first half of the vectors
	nt := nb / 2
	err = index.Train(xb[:nt*d])
	if err != nil {
		t.Fatalf("Train failed: %v", err)
	}
	if !index.IsTrained() {
		t.Fatal("expected index to be trained")
	}

	// 3. Add vectors
	err = index.Add(xb)
	if err != nil {
		t.Fatalf("Add failed: %v", err)
	}
	if index.Ntotal() != int64(nb) {
		t.Fatalf("expected %d vectors, got %d", nb, index.Ntotal())
	}

	// Search before saving
	distances1, labels1, err := index.Search(xq, k)
	if err != nil {
		t.Fatalf("Search before saving failed: %v", err)
	}

	// 4. Write index to file
	tmpdir := t.TempDir()
	fname := filepath.Join(tmpdir, "test_ivf.index")
	err = index.Write(fname)
	if err != nil {
		t.Fatalf("Write failed: %v", err)
	}

	// 5. Read index from file
	loadedIndex, err := ReadIndex(fname)
	if err != nil {
		t.Fatalf("ReadIndex failed: %v", err)
	}
	defer loadedIndex.Free()

	// 6. Verify loaded index
	if loadedIndex.D() != d {
		t.Errorf("loaded index has wrong dimension: got %d, want %d", loadedIndex.D(), d)
	}
	if !loadedIndex.IsTrained() {
		t.Error("loaded index should be trained")
	}
	if loadedIndex.Ntotal() != int64(nb) {
		t.Errorf("loaded index has wrong ntotal: got %d, want %d", loadedIndex.Ntotal(), int64(nb))
	}

	// 7. Search after loading and compare results
	distances2, labels2, err := loadedIndex.Search(xq, k)
	if err != nil {
		t.Fatalf("Search after loading failed: %v", err)
	}

	if len(distances1) != len(distances2) || len(labels1) != len(labels2) {
		t.Fatalf("search results have different lengths")
	}

	for i := range labels1 {
		if labels1[i] != labels2[i] {
			t.Fatalf("labels mismatch at index %d: original %d, loaded %d", i, labels1[i], labels2[i])
		}
	}

	t.Log("Successfully created, trained, saved, loaded, and searched an IVF,Flat index.")
}

func ExampleIndex_workflow() {
	// This example demonstrates the full workflow:
	// 1. Create an IVF (Inverted File) index.
	// 2. Train it with some data.
	// 3. Add data to the index.
	// 4. Save the index to a file.
	// 5. Load the index from the file.
	// 6. Perform a search.

	// Use a fixed seed to make the example deterministic for both Go and Faiss.
	rand.Seed(42)

	d := 16      // vector dimension
	nb := 1000   // database size
	nlist := 10  // ivf clusters (reduced to avoid Faiss warning)
	k := 4       // neighbors to search
	nt := nb / 2 // training data size
	nq := 10     // query data size

	// Generate some random data
	xb := make([]float32, d*nb)
	for i := range xb {
		xb[i] = rand.Float32()
	}
	xq := make([]float32, d*nq)
	for i := range xq {
		xq[i] = rand.Float32()
	}

	// 1. Create
	index, err := NewIndex(d, fmt.Sprintf("IVF%d,Flat", nlist), MetricL2)
	if err != nil {
		fmt.Printf("Failed to create index: %v\n", err)
		return
	}
	defer index.Free()

	// 2. Train
	if err := index.Train(xb[:nt*d]); err != nil {
		fmt.Printf("Failed to train index: %v\n", err)
		return
	}

	// 3. Add
	if err := index.Add(xb); err != nil {
		fmt.Printf("Failed to add vectors: %v\n", err)
		return
	}
	fmt.Printf("Index is trained: %t, Ntotal: %d\n", index.IsTrained(), index.Ntotal())

	// 4. Save
	tmpfile, err := os.CreateTemp("", "example-*.index")
	if err != nil {
		fmt.Printf("Failed to create temp file: %v\n", err)
		return
	}
	defer os.Remove(tmpfile.Name())
	if err := index.Write(tmpfile.Name()); err != nil {
		fmt.Printf("Failed to write index: %v\n", err)
		return
	}

	// 5. Load
	loadedIndex, err := ReadIndex(tmpfile.Name())
	if err != nil {
		fmt.Printf("Failed to read index: %v\n", err)
		return
	}
	defer loadedIndex.Free()
	fmt.Printf("Loaded index is trained: %t, Ntotal: %d\n", loadedIndex.IsTrained(), loadedIndex.Ntotal())

	// 6. Search
	_, labels, err := loadedIndex.Search(xq, k)
	if err != nil {
		fmt.Printf("Search failed: %v\n", err)
		return
	}

	fmt.Printf("Search complete. Example results for first query:\n")
	for i := 0; i < k; i++ {
		fmt.Printf("  - Neighbor %d: Label %d\n", i+1, labels[i])
	}

	// Output:
	// Index is trained: true, Ntotal: 1000
	// Loaded index is trained: true, Ntotal: 1000
	// Search complete. Example results for first query:
	//   - Neighbor 1: Label 451
	//   - Neighbor 2: Label 152
	//   - Neighbor 3: Label 482
	//   - Neighbor 4: Label 531
}
