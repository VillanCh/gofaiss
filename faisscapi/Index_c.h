/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c -*-

#ifndef FAISS_INDEX_C_H
#define FAISS_INDEX_C_H

#include <stddef.h>
#include <faiss_c.h>

#ifdef __cplusplus
extern "C" {
#endif

// forward declaration required here
FAISS_DECLARE_CLASS(RangeSearchResult)

// typedef struct FaissRangeSearchResult_H FaissRangeSearchResult;
typedef struct FaissIDSelector_H FaissIDSelector;

/// Some algorithms support both an inner product version and a L2 search
/// version.
typedef enum FaissMetricType {
    METRIC_INNER_PRODUCT = 0, ///< maximum inner product search
    METRIC_L2 = 1,            ///< squared L2 search
    METRIC_L1,                ///< L1 (aka cityblock)
    METRIC_Linf,              ///< infinity distance
    METRIC_Lp,                ///< L_p distance, p is given by metric_arg

    /// some additional metrics defined in scipy.spatial.distance
    METRIC_Canberra = 20,
    METRIC_BrayCurtis,
    METRIC_JensenShannon,
} FaissMetricType;

FAISS_DECLARE_CLASS(SearchParameters)
FAISS_DECLARE_DESTRUCTOR(SearchParameters)

int faiss_SearchParameters_new(
        FaissSearchParameters** p_sp,
        FaissIDSelector* sel);

/// Opaque type for referencing to an index object
FAISS_DECLARE_CLASS(Index)
FAISS_DECLARE_DESTRUCTOR(Index)

/// Getter for d
FAISS_DECLARE_GETTER(Index, int, d)

/// Getter for is_trained
FAISS_DECLARE_GETTER(Index, int, is_trained)

/// Getter for ntotal
FAISS_DECLARE_GETTER(Index, idx_t, ntotal)

/// Getter for metric_type
FAISS_DECLARE_GETTER(Index, FaissMetricType, metric_type)

FAISS_DECLARE_GETTER_SETTER(Index, int, verbose)

/** Perform training on a representative set of vectors
 *
 * @param index  opaque pointer to index object
 * @param n      nb of training vectors
 * @param x      training vectors, size n * d
 */
int faiss_Index_train(FaissIndex* index, idx_t n, const float* x);

/** Add n vectors of dimension d to the index.
 *
 * Vectors are implicitly assigned labels ntotal .. ntotal + n - 1
 * This function slices the input vectors in chunks smaller than
 * blocksize_add and calls add_core.
 * @param index  opaque pointer to index object
 * @param x      input matrix, size n * d
 */
int faiss_Index_add(FaissIndex* index, idx_t n, const float* x);

/** Same as add, but stores xids instead of sequential ids.
 *
 * The default implementation fails with an assertion, as it is
 * not supported by all indexes.
 *
 * @param index  opaque pointer to index object
 * @param xids   if non-null, ids to store for the vectors (size n)
 */
int faiss_Index_add_with_ids(
        FaissIndex* index,
        idx_t n,
        const float* x,
        const idx_t* xids);

/** query n vectors of dimension d to the index.
 *
 * return at most k vectors. If there are not enough results for a
 * query, the result array is padded with -1s.
 *
 * @param index       opaque pointer to index object
 * @param x           input vectors to search, size n * d
 * @param labels      output labels of the NNs, size n*k
 * @param distances   output pairwise distances, size n*k
 */
int faiss_Index_search(
        const FaissIndex* index,
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels);

/**
 * query n vectors of dimension d with search parameters to the index.
 *
 * return at most k vectors. If there are not enough results for a query,
 * the result is padded with -1s.
 *
 * @param index       opaque pointer to index object
 * @param x           input vectors to search, size n * d
 * @param params      input params to modify how search is done
 * @param labels      output labels of the NNs, size n*k
 * @param distances   output pairwise distances, size n*k
 */
int faiss_Index_search_with_params(
        const FaissIndex* index,
        idx_t n,
        const float* x,
        idx_t k,
        const FaissSearchParameters* params,
        float* distances,
        idx_t* labels);

/** query n vectors of dimension d to the index.
 *
 * return all vectors with distance < radius. Note that many
 * indexes do not implement the range_search (only the k-NN search
 * is mandatory).
 *
 * @param index       opaque pointer to index object
 * @param x           input vectors to search, size n * d
 * @param radius      search radius
 * @param result      result table
 */
int faiss_Index_range_search(
        const FaissIndex* index,
        idx_t n,
        const float* x,
        float radius,
        FaissRangeSearchResult* result);

/** return the indexes of the k vectors closest to the query x.
 *
 * This function is identical as search but only return labels of neighbors.
 * @param index       opaque pointer to index object
 * @param x           input vectors to search, size n * d
 * @param labels      output labels of the NNs, size n*k
 */
int faiss_Index_assign(
        FaissIndex* index,
        idx_t n,
        const float* x,
        idx_t* labels,
        idx_t k);

/** removes all elements from the database.
 * @param index       opaque pointer to index object
 */
int faiss_Index_reset(FaissIndex* index);

/** removes IDs from the index. Not supported by all indexes
 * @param index       opaque pointer to index object
 * @param nremove     output for the number of IDs removed
 */
int faiss_Index_remove_ids(
        FaissIndex* index,
        const FaissIDSelector* sel,
        size_t* n_removed);

/** Reconstruct a stored vector (or an approximation if lossy coding)
 *
 * this function may not be defined for some indexes
 * @param index       opaque pointer to index object
 * @param key         id of the vector to reconstruct
 * @param recons      reconstructed vector (size d)
 */
int faiss_Index_reconstruct(const FaissIndex* index, idx_t key, float* recons);

/** Reconstruct vectors i0 to i0 + ni - 1
 *
 * this function may not be defined for some indexes
 * @param index       opaque pointer to index object
 * @param recons      reconstructed vector (size ni * d)
 */
int faiss_Index_reconstruct_n(
        const FaissIndex* index,
        idx_t i0,
        idx_t ni,
        float* recons);

/** Computes a residual vector after indexing encoding.
 *
 * The residual vector is the difference between a vector and the
 * reconstruction that can be decoded from its representation in
 * the index. The residual can be used for multiple-stage indexing
 * methods, like IndexIVF's methods.
 *
 * @param index       opaque pointer to index object
 * @param x           input vector, size d
 * @param residual    output residual vector, size d
 * @param key         encoded index, as returned by search and assign
 */
int faiss_Index_compute_residual(
        const FaissIndex* index,
        const float* x,
        float* residual,
        idx_t key);

/** Computes a residual vector after indexing encoding.
 *
 * The residual vector is the difference between a vector and the
 * reconstruction that can be decoded from its representation in
 * the index. The residual can be used for multiple-stage indexing
 * methods, like IndexIVF's methods.
 *
 * @param index       opaque pointer to index object
 * @param n           number of vectors
 * @param x           input vector, size (n x d)
 * @param residuals    output residual vectors, size (n x d)
 * @param keys         encoded index, as returned by search and assign
 */
int faiss_Index_compute_residual_n(
        const FaissIndex* index,
        idx_t n,
        const float* x,
        float* residuals,
        const idx_t* keys);

/* The standalone codec interface */

/** The size of the produced codes in bytes.
 *
 * @param index   opaque pointer to index object
 * @param size    the returned size in bytes
 */
int faiss_Index_sa_code_size(const FaissIndex* index, size_t* size);

/** encode a set of vectors
 *
 * @param index   opaque pointer to index object
 * @param n       number of vectors
 * @param x       input vectors, size n * d
 * @param bytes   output encoded vectors, size n * sa_code_size()
 */
int faiss_Index_sa_encode(
        const FaissIndex* index,
        idx_t n,
        const float* x,
        uint8_t* bytes);

/** decode a set of vectors
 *
 * @param index   opaque pointer to index object
 * @param n       number of vectors
 * @param bytes   input encoded vectors, size n * sa_code_size()
 * @param x       output vectors, size n * d
 */
int faiss_Index_sa_decode(
        const FaissIndex* index,
        idx_t n,
        const uint8_t* bytes,
        float* x);

#ifdef __cplusplus
}
#endif

#endif
