/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-
// I/O code for indexes

#ifndef FAISS_CLONE_INDEX_C_H
#define FAISS_CLONE_INDEX_C_H

#include <stdio.h>
#include <IndexBinary_c.h>
#include <Index_c.h>
#include <faiss_c.h>

#ifdef __cplusplus
extern "C" {
#endif

/* cloning functions */

/** Clone an index. This is equivalent to `faiss::clone_index` */
int faiss_clone_index(const FaissIndex*, FaissIndex** p_out);

/** Clone a binary index. This is equivalent to `faiss::clone_index_binary` */
int faiss_clone_index_binary(const FaissIndexBinary*, FaissIndexBinary** p_out);

#ifdef __cplusplus
}
#endif
#endif
