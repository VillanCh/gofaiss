//go:build darwin

package gofaiss

/*
#cgo CFLAGS: -I${SRCDIR}/faisscapi
#cgo LDFLAGS: -L${SRCDIR}/faisslib/v1.11.0/darwin -lfaiss_c -lfaiss
#cgo LDFLAGS: -Wl,-rpath,${SRCDIR}/faisslib/v1.11.0/darwin
*/
import "C"
