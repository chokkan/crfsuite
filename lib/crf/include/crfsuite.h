/*
 *      CRFsuite library.
 *
 * Copyright (c) 2007-2010, Naoaki Okazaki
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the names of the authors nor the names of its contributors
 *       may be used to endorse or promote products derived from this
 *       software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* $Id$ */

#ifndef    __CRFSUITE_H__
#define    __CRFSUITE_H__

#ifdef    __cplusplus
extern "C" {
#endif/*__cplusplus*/

#include <stdio.h>
#include <stdarg.h>

/* Forward declarations */
struct tag_crf_model;
typedef struct tag_crf_model crf_model_t;

struct tag_crf_trainer;
typedef struct tag_crf_trainer crf_trainer_t;

struct tag_crf_tagger;
typedef struct tag_crf_tagger crf_tagger_t;

struct tag_crf_dictionary;
typedef struct tag_crf_dictionary crf_dictionary_t;

struct tag_crf_params;
typedef struct tag_crf_params crf_params_t;

enum {
    CRF_SUCCESS = 0,
    CRFERR_UNKNOWN = 0x80000000,
    CRFERR_OUTOFMEMORY,
    CRFERR_NOTSUPPORTED,
    CRFERR_INCOMPATIBLE,
    CRFERR_INTERNAL_LOGIC,
    CRFERR_OVERFLOW,
};


/**
 * Content of an item.
 *    A content consists of an attribute id with its frequency in the item.
 */
typedef struct {
    int            aid;    /**< Attribute id. */
    floatval_t    scale;    /**< Scale factor (frequency) of the attribute. */
} crf_content_t;

/**
 * An item.
 */
typedef struct {
    int                num_contents;    /**< Number of contents associated with the item. */
    int                max_contents;    /**< Maximum number of contents. */
    crf_content_t*    contents;        /**< Array of the contents. */
    int                label;            /**< Output label. */
} crf_item_t;

/**
 * A sequence.
 */
typedef struct {
    int            num_items;    /**< Number of items in the sequence. */
    int            max_items;    /**< Maximum number of items (internal use). */
    crf_item_t*    items;        /**< Array of the items. */
} crf_sequence_t;

/**
 * A data.
 */
typedef struct {
    int                num_instances;        /**< Number of instances. */
    int                max_instances;        /**< Maximum number of instances (internal use). */
    int                num_labels;            /**< Number of distinct labels. */
    int                num_attrs;            /**< Number of distinct attributes. */
    int                max_item_length;    /**< Maximum item length. */
    crf_sequence_t*    instances;            /**< Array of instances. */
} crf_data_t;

/**
 * An output label sequence.
 */
typedef struct {
    int        num_labels;            /**< Number of output labels. */
    int*    labels;                /**< Array of the output labels. */
    floatval_t    probability;        /**< Probability of the output labels. */
} crf_output_t;

typedef struct {
    int        num_correct;
    int        num_observation;
    int        num_model;
    int        num_total;
    floatval_t    precision;
    floatval_t    recall;
    floatval_t    fmeasure;
} crf_label_evaluation_t;

typedef struct {
    int        num_labels;
    crf_label_evaluation_t* tbl;

    int        item_total_correct;
    int     item_total_num;
    int        item_total_model;
    int        item_total_observation;
    floatval_t    item_accuracy;

    int        inst_total_correct;
    int        inst_total_num;
    floatval_t    inst_accuracy;

    floatval_t    macro_precision;
    floatval_t    macro_recall;
    floatval_t    macro_fmeasure;
} crf_evaluation_t;




typedef int (*crf_logging_callback)(void *instance, const char *format, va_list args);
typedef int (*crf_evaluate_callback)(void *instance, crf_tagger_t* tagger);


struct tag_crf_model {
    /**
     * Pointer to the instance data (internal use only).
     */
    void *internal;
    
    /**
     * Reference counter (internal use only).
     */
    int nref;

    /**
     * Increment the reference counter.
     */
    int (*addref)(crf_model_t* model);

    /**
     * Decrement the reference counter.
     */
    int (*release)(crf_model_t* model);

    int (*get_tagger)(crf_model_t* model, crf_tagger_t** ptr_tagger);
    int (*get_labels)(crf_model_t* model, crf_dictionary_t** ptr_labels);
    int (*get_attrs)(crf_model_t* model, crf_dictionary_t** ptr_attrs);
    int (*dump)(crf_model_t* model, FILE *fpo);
};



struct tag_crf_trainer {
    /**
     * Pointer to the instance data (internal use only).
     */
    void *internal;
    
    /**
     * Reference counter (internal use only).
     */
    int nref;

    /**
     * Increment the reference counter.
     */
    int (*addref)(crf_trainer_t* trainer);

    /**
     * Decrement the reference counter.
     */
    int (*release)(crf_trainer_t* trainer);

    crf_params_t* (*params)(crf_trainer_t* trainer);

    void (*set_message_callback)(crf_trainer_t* trainer, void *instance, crf_logging_callback cbm);
    void (*set_evaluate_callback)(crf_trainer_t* trainer, void *instance, crf_evaluate_callback cbe);

    int (*train)(crf_trainer_t* trainer, void* instances, int num_instances, int num_labels, int num_attributes);
    int (*save)(crf_trainer_t* trainer, const char *filename, crf_dictionary_t* attrs, crf_dictionary_t* labels);
};

struct tag_crf_tagger {
    /**
     * Pointer to the instance data (internal use only).
     */
    void *internal;

    /**
     * Reference counter (internal use only).
     */
    int nref;

    /**
     * Increment the reference counter.
     */
    int (*addref)(crf_tagger_t* tagger);

    /**
     * Decrement the reference counter.
     */
    int (*release)(crf_tagger_t* tagger);

    /**
     * Tag an input sequence.
     */
    int (*tag)(crf_tagger_t* tagger, crf_sequence_t *inst, crf_output_t* output);

};

struct tag_crf_dictionary {
    /**
     * Pointer to the instance data (internal use only).
     */
    void *internal;

    /**
     * Reference counter (internal use only).
     */
    int nref;

    /**
     * Increment the reference counter.
     */
    int (*addref)(crf_dictionary_t* dic);

    /**
     * Decrement the reference counter.
     */
    int (*release)(crf_dictionary_t* dic);

    int (*get)(crf_dictionary_t* dic, const char *str);
    int (*to_id)(crf_dictionary_t* dic, const char *str);
    int (*to_string)(crf_dictionary_t* dic, int id, char const **pstr);
    int (*num)(crf_dictionary_t* dic);
    void (*free)(crf_dictionary_t* dic, const char *str);
};

struct tag_crf_params {
    /**
     * Pointer to the instance data (internal use only).
     */
    void *internal;

    /**
     * Reference counter (internal use only).
     */
    int nref;

    /**
     * Increment the reference counter.
     */
    int (*addref)(crf_params_t* params);

    /**
     * Decrement the reference counter.
     */
    int (*release)(crf_params_t* params);

    int (*set)(crf_params_t* params, const char *name, const char *value);
    int (*set_int)(crf_params_t* params, const char *name, int value);
    int (*set_float)(crf_params_t* params, const char *name, floatval_t value);
    int (*set_string)(crf_params_t* params, const char *name, const char *value);

    int (*get_int)(crf_params_t* params, const char *name, int *value);
    int (*get_float)(crf_params_t* params, const char *name, floatval_t *value);
    int (*get_string)(crf_params_t* params, const char *name, char **value);
};



int crf_create_instance(const char *iid, void **ptr);
int crf_create_instance_from_file(const char *filename, void **ptr);

int crf_create_tagger(
    const char *filename,
    crf_tagger_t** ptr_tagger,
    crf_dictionary_t** ptr_attrs,
    crf_dictionary_t** ptr_labels
    );


void crf_content_init(crf_content_t* cont);
void crf_content_set(crf_content_t* cont, int aid, floatval_t scale);
void crf_content_copy(crf_content_t* dst, const crf_content_t* src);
void crf_content_swap(crf_content_t* x, crf_content_t* y);

void crf_item_init(crf_item_t* item);
void crf_item_init_n(crf_item_t* item, int num_contents);
void crf_item_finish(crf_item_t* item);
void crf_item_copy(crf_item_t* dst, const crf_item_t* src);
void crf_item_swap(crf_item_t* x, crf_item_t* y);
int  crf_item_append_content(crf_item_t* item, const crf_content_t* cont);
int  crf_item_empty(crf_item_t* item);

void crf_sequence_init(crf_sequence_t* seq);
void crf_sequence_init_n(crf_sequence_t* seq, int num_items);
void crf_sequence_finish(crf_sequence_t* seq);
void crf_sequence_copy(crf_sequence_t* dst, const crf_sequence_t* src);
void crf_sequence_swap(crf_sequence_t* x, crf_sequence_t* y);
int  crf_sequence_append(crf_sequence_t* seq, const crf_item_t* item, int label);
int  crf_sequence_empty(crf_sequence_t* seq);

void crf_data_init(crf_data_t* data);
void crf_data_init_n(crf_data_t* data, int n);
void crf_data_finish(crf_data_t* data);
void crf_data_copy(crf_data_t* dst, const crf_data_t* src);
void crf_data_swap(crf_data_t* x, crf_data_t* y);
int  crf_data_append(crf_data_t* data, const crf_sequence_t* inst);
int  crf_data_maxlength(crf_data_t* data);
int  crf_data_totalitems(crf_data_t* data);

void crf_output_init(crf_output_t* output);
void crf_output_init_n(crf_output_t* output, int n);
void crf_output_finish(crf_output_t* outpu);

void crf_evaluation_init(crf_evaluation_t* eval, int n);
void crf_evaluation_finish(crf_evaluation_t* eval);
void crf_evaluation_clear(crf_evaluation_t* eval);
int crf_evaluation_accmulate(crf_evaluation_t* eval, const crf_sequence_t* reference, const crf_output_t* target);
void crf_evaluation_compute(crf_evaluation_t* eval);
void crf_evaluation_output(crf_evaluation_t* eval, crf_dictionary_t* labels, FILE *fpo);


int crf_interlocked_increment(int *count);
int crf_interlocked_decrement(int *count);


#ifdef    __cplusplus
}
#endif/*__cplusplus*/

#endif/*__CRFSUITE_H__*/
