# GoFaiss C-API Wrapper

This directory contains the necessary files to use Faiss via its C API in a Go application. This package is self-contained and includes the compiled dynamic libraries.

## Directory Structure

- `gofaiss/`: The Go wrapper package.
  - `gofaiss.go`: The main Go source file with CGO bindings for `darwin`.
  - `gofaiss_test.go`: Test file for the package.
  - `faisscapi/`: Contains all the necessary C header files for Faiss C API.
  - `faisslib/`: Contains the compiled Faiss dynamic libraries.
    - `v1.11.0/`: Versioned directory.
      - `darwin/`: For macOS.
        - `libfaiss.dylib`
        - `libfaiss_c.dylib`
      - `linux/`: (placeholder for `.so` files)
      - `windows/`: (placeholder for `.dll` files)
- `include/`: (Source) Contains the C header files used to build the package.
- `go.mod`: Go module file.
- `README.md`: This file.

## Prerequisites

- Go installed.
- A C compiler (like Clang or GCC).

## How to Test

Navigate to the `gofaiss` directory and run the tests:

```bash
cd faiss_c_api_lib/gofaiss
go test -v
```

This will build the Go package, linking against the dynamic libraries included in the `gofaiss/faisslib` directory for your specific OS. The Go build tag system is used to select the correct library path.

The `-rpath` setting in the `LDFLAGS` (inside `gofaiss.go`) helps the Go executable find the dynamic libraries at runtime without needing to set environment variables like `LD_LIBRARY_PATH` or `DYLD_LIBRARY_PATH`.

## How to Use

To use this package in your own Go project, you can copy the `gofaiss` directory into your project.

Example:

```go
package main

import (
    "fmt"
    "log"

    "path/to/your/project/gofaiss"
)

func main() {
    d := 128 // dimension
    index, err := gofaiss.NewIndexFlatL2(d)
    if err != nil {
        log.Fatalf("Failed to create index: %v", err)
    }
    defer index.Free()

    fmt.Printf("Index created successfully. Dimension: %d\n", index.D())
    // ... add vectors and search ...
}
```

Enjoy using Faiss in Go! 