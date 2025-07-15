package gofaiss

import (
	"math/rand"
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
