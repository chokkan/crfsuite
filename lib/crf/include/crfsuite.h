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

typedef double floatval_t;
#define    FLOAT_MAX    DBL_MAX

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

/**
 * Status codes.
 */
enum {
    CRF_SUCCESS = 0,
    CRFERR_UNKNOWN = 0x80000000,
    CRFERR_OUTOFMEMORY,
    CRFERR_NOTSUPPORTED,
    CRFERR_INCOMPATIBLE,
    CRFERR_INTERNAL_LOGIC,
    CRFERR_OVERFLOW,
    CRFERR_NOTIMPLEMENTED,
};


/**
 * An attribute content.
 *  An attribute content consists of an attribute id with its weight (frequency).
 */
typedef struct {
    int         aid;                /**< Attribute id. */
    floatval_t  scale;              /**< Weight (frequency) of the attribute. */
} crf_content_t;

/**
 * An item.
 *  An item consists of an array of attribute contents.
 */
typedef struct {
    int             num_contents;   /**< Number of contents associated with the item. */
    int             cap_contents;   /**< Maximum number of contents (internal use). */
    crf_content_t   *contents;      /**< Array of the contents. */
} crf_item_t;

/**
 * An instance (sequence of items and labels).
 *  An instance consists of a sequence of items and labels.
 */
typedef struct {
    int         num_items;          /**< Number of items/labels in the sequence. */
    int         cap_items;          /**< Maximum number of items/labels (internal use). */
    crf_item_t  *items;             /**< Array of the item sequence. */
    int         *labels;            /**< Array of the label sequence. */
	int         group;              /**< Group ID of the instance. */
} crf_instance_t;

/**
 * A data set.
 */
typedef struct {
    int                 num_instances;        /**< Number of instances. */
    int                 cap_instances;        /**< Maximum number of instances (internal use). */
    crf_instance_t*     instances;            /**< Array of instances. */

    crf_dictionary_t    *attrs;
    crf_dictionary_t    *labels;
} crf_data_t;

/**
 * A label-wise performance values.
 */
typedef struct {
    int         num_correct;        /**< Number of correct predictions. */
    int         num_observation;    /**< Number of predictions. */
    int         num_model;          /**< Number of occurrences in the gold-standard data. */
    int         num_total;          /**< */
    floatval_t  precision;          /**< Precision. */
    floatval_t  recall;             /**< Recall. */
    floatval_t  fmeasure;           /**< F1 score. */
} crf_label_evaluation_t;

/**
 * An overall performance values.
 */
typedef struct {
    int         num_labels;         /**< Number of labels. */
    crf_label_evaluation_t* tbl;    /**< Array of label-wise evaluations. */

    int         item_total_correct;
    int         item_total_num;
    int         item_total_model;
    int         item_total_observation;
    floatval_t  item_accuracy;      /**< Item accuracy. */

    int         inst_total_correct; /**< Number of correctly predicted instances. */
    int         inst_total_num;     /**< Total number of instances. */
    floatval_t  inst_accuracy;      /**< Instance accuracy. */

    floatval_t  macro_precision;    /**< Macro-averaged precision. */
    floatval_t  macro_recall;       /**< Macro-averaged recall. */
    floatval_t  macro_fmeasure;     /**< Macro-averaged F1 score. */
} crf_evaluation_t;




typedef int (*crf_logging_callback)(void *instance, const char *format, va_list args);


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

    int (*train)(crf_trainer_t* trainer, const crf_data_t *data, const char *filename, int holdout);
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
    int (*tag)(crf_tagger_t* tagger, crf_instance_t *inst, int *labels, floatval_t *ptr_score);

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

void crf_instance_init(crf_instance_t* seq);
void crf_instance_init_n(crf_instance_t* seq, int num_items);
void crf_instance_finish(crf_instance_t* seq);
void crf_instance_copy(crf_instance_t* dst, const crf_instance_t* src);
void crf_instance_swap(crf_instance_t* x, crf_instance_t* y);
int  crf_instance_append(crf_instance_t* seq, const crf_item_t* item, int label);
int  crf_instance_empty(crf_instance_t* seq);

void crf_data_init(crf_data_t* data);
void crf_data_init_n(crf_data_t* data, int n);
void crf_data_finish(crf_data_t* data);
void crf_data_copy(crf_data_t* dst, const crf_data_t* src);
void crf_data_swap(crf_data_t* x, crf_data_t* y);
int  crf_data_append(crf_data_t* data, const crf_instance_t* inst);
int  crf_data_maxlength(crf_data_t* data);
int  crf_data_totalitems(crf_data_t* data);

void crf_evaluation_init(crf_evaluation_t* eval, int n);
void crf_evaluation_finish(crf_evaluation_t* eval);
void crf_evaluation_clear(crf_evaluation_t* eval);
int crf_evaluation_accmulate(crf_evaluation_t* eval, const crf_instance_t* reference, const int* target);
void crf_evaluation_compute(crf_evaluation_t* eval);
void crf_evaluation_output(crf_evaluation_t* eval, crf_dictionary_t* labels, crf_logging_callback cbm, void *instance);


int crf_interlocked_increment(int *count);
int crf_interlocked_decrement(int *count);


#ifdef    __cplusplus
}
#endif/*__cplusplus*/

#endif/*__CRFSUITE_H__*/
