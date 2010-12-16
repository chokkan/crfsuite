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

#define CRFSUITE_VERSION    "0.11.2"
#define CRFSUITE_COPYRIGHT  "Copyright (c) 2007-2010 Naoaki Okazaki"

typedef double floatval_t;
#define    FLOAT_MAX    DBL_MAX

/* Forward declarations */
struct tag_crfsuite_model;
typedef struct tag_crfsuite_model crfsuite_model_t;

struct tag_crfsuite_trainer;
typedef struct tag_crfsuite_trainer crfsuite_trainer_t;

struct tag_crfsuite_tagger;
typedef struct tag_crfsuite_tagger crfsuite_tagger_t;

struct tag_crfsuite_dictionary;
typedef struct tag_crfsuite_dictionary crfsuite_dictionary_t;

struct tag_crfsuite_params;
typedef struct tag_crfsuite_params crfsuite_params_t;

/**
 * Status codes.
 */
enum {
    CRFSUITE_SUCCESS = 0,
    CRFSUITEERR_UNKNOWN = 0x80000000,
    CRFSUITEERR_OUTOFMEMORY,
    CRFSUITEERR_NOTSUPPORTED,
    CRFSUITEERR_INCOMPATIBLE,
    CRFSUITEERR_INTERNAL_LOGIC,
    CRFSUITEERR_OVERFLOW,
    CRFSUITEERR_NOTIMPLEMENTED,
};


/**
 * An attribute content.
 *  An attribute content consists of an attribute id with its weight (frequency).
 */
typedef struct {
    int         aid;                /**< Attribute id. */
    floatval_t  scale;              /**< Weight (frequency) of the attribute. */
} crfsuite_content_t;

/**
 * An item.
 *  An item consists of an array of attribute contents.
 */
typedef struct {
    int             num_contents;   /**< Number of contents associated with the item. */
    int             cap_contents;   /**< Maximum number of contents (internal use). */
    crfsuite_content_t   *contents;      /**< Array of the contents. */
} crfsuite_item_t;

/**
 * An instance (sequence of items and labels).
 *  An instance consists of a sequence of items and labels.
 */
typedef struct {
    int         num_items;          /**< Number of items/labels in the sequence. */
    int         cap_items;          /**< Maximum number of items/labels (internal use). */
    crfsuite_item_t  *items;             /**< Array of the item sequence. */
    int         *labels;            /**< Array of the label sequence. */
	int         group;              /**< Group ID of the instance. */
} crfsuite_instance_t;

/**
 * A data set.
 */
typedef struct {
    int                 num_instances;        /**< Number of instances. */
    int                 cap_instances;        /**< Maximum number of instances (internal use). */
    crfsuite_instance_t*     instances;            /**< Array of instances. */

    crfsuite_dictionary_t    *attrs;
    crfsuite_dictionary_t    *labels;
} crfsuite_data_t;

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
} crfsuite_label_evaluation_t;

/**
 * An overall performance values.
 */
typedef struct {
    int         num_labels;         /**< Number of labels. */
    crfsuite_label_evaluation_t* tbl;    /**< Array of label-wise evaluations. */

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
} crfsuite_evaluation_t;




typedef int (*crfsuite_logging_callback)(void *instance, const char *format, va_list args);


struct tag_crfsuite_model {
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
    int (*addref)(crfsuite_model_t* model);

    /**
     * Decrement the reference counter.
     */
    int (*release)(crfsuite_model_t* model);

    int (*get_tagger)(crfsuite_model_t* model, crfsuite_tagger_t** ptr_tagger);
    int (*get_labels)(crfsuite_model_t* model, crfsuite_dictionary_t** ptr_labels);
    int (*get_attrs)(crfsuite_model_t* model, crfsuite_dictionary_t** ptr_attrs);
    int (*dump)(crfsuite_model_t* model, FILE *fpo);
};



struct tag_crfsuite_trainer {
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
    int (*addref)(crfsuite_trainer_t* trainer);

    /**
     * Decrement the reference counter.
     */
    int (*release)(crfsuite_trainer_t* trainer);

    crfsuite_params_t* (*params)(crfsuite_trainer_t* trainer);

    void (*set_message_callback)(crfsuite_trainer_t* trainer, void *instance, crfsuite_logging_callback cbm);

    int (*train)(crfsuite_trainer_t* trainer, const crfsuite_data_t *data, const char *filename, int holdout);
};

struct tag_crfsuite_tagger {
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
    int (*addref)(crfsuite_tagger_t* tagger);

    /**
     * Decrement the reference counter.
     */
    int (*release)(crfsuite_tagger_t* tagger);

    /**
     * Sets an instance.
     */
    int (*set)(crfsuite_tagger_t* tagger, crfsuite_instance_t *inst);

    int (*length)(crfsuite_tagger_t* tagger);

    /**
     * Obtains the Viterbi label sequence.
     */
    int (*viterbi)(crfsuite_tagger_t* tagger, int *labels, floatval_t *ptr_score);
    int (*score)(crfsuite_tagger_t* tagger, int *path, floatval_t *ptr_score);

    int (*lognorm)(crfsuite_tagger_t* tagger, floatval_t *ptr_norm);

    int (*marginal_point)(crfsuite_tagger_t *tagger, int label, int t, floatval_t *ptr_prob);
    int (*marginal_path)(crfsuite_tagger_t *tagger, const int *path, int begin, int end, floatval_t *ptr_prob);
};

struct tag_crfsuite_dictionary {
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
    int (*addref)(crfsuite_dictionary_t* dic);

    /**
     * Decrement the reference counter.
     */
    int (*release)(crfsuite_dictionary_t* dic);

    int (*get)(crfsuite_dictionary_t* dic, const char *str);
    int (*to_id)(crfsuite_dictionary_t* dic, const char *str);
    int (*to_string)(crfsuite_dictionary_t* dic, int id, char const **pstr);
    int (*num)(crfsuite_dictionary_t* dic);
    void (*free)(crfsuite_dictionary_t* dic, const char *str);
};

struct tag_crfsuite_params {
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
    int (*addref)(crfsuite_params_t* params);

    /**
     * Decrement the reference counter.
     */
    int (*release)(crfsuite_params_t* params);

    int (*num)(crfsuite_params_t* params);
    int (*name)(crfsuite_params_t* params, int i, char **ptr_name);

    int (*set)(crfsuite_params_t* params, const char *name, const char *value);
    int (*get)(crfsuite_params_t* params, const char *name, char **ptr_value);
    void (*free)(crfsuite_params_t* params, const char *str);

    int (*set_int)(crfsuite_params_t* params, const char *name, int value);
    int (*set_float)(crfsuite_params_t* params, const char *name, floatval_t value);
    int (*set_string)(crfsuite_params_t* params, const char *name, const char *value);

    int (*get_int)(crfsuite_params_t* params, const char *name, int *value);
    int (*get_float)(crfsuite_params_t* params, const char *name, floatval_t *value);
    int (*get_string)(crfsuite_params_t* params, const char *name, char **ptr_value);

    int (*help)(crfsuite_params_t* params, const char *name, char **ptr_type, char **ptr_help);
};



int crfsuite_create_instance(const char *iid, void **ptr);
int crfsuite_create_instance_from_file(const char *filename, void **ptr);

int crfsuite_create_tagger(
    const char *filename,
    crfsuite_tagger_t** ptr_tagger,
    crfsuite_dictionary_t** ptr_attrs,
    crfsuite_dictionary_t** ptr_labels
    );


void crfsuite_content_init(crfsuite_content_t* cont);
void crfsuite_content_set(crfsuite_content_t* cont, int aid, floatval_t scale);
void crfsuite_content_copy(crfsuite_content_t* dst, const crfsuite_content_t* src);
void crfsuite_content_swap(crfsuite_content_t* x, crfsuite_content_t* y);

void crfsuite_item_init(crfsuite_item_t* item);
void crfsuite_item_init_n(crfsuite_item_t* item, int num_contents);
void crfsuite_item_finish(crfsuite_item_t* item);
void crfsuite_item_copy(crfsuite_item_t* dst, const crfsuite_item_t* src);
void crfsuite_item_swap(crfsuite_item_t* x, crfsuite_item_t* y);
int  crfsuite_item_append_content(crfsuite_item_t* item, const crfsuite_content_t* cont);
int  crfsuite_item_empty(crfsuite_item_t* item);

void crfsuite_instance_init(crfsuite_instance_t* seq);
void crfsuite_instance_init_n(crfsuite_instance_t* seq, int num_items);
void crfsuite_instance_finish(crfsuite_instance_t* seq);
void crfsuite_instance_copy(crfsuite_instance_t* dst, const crfsuite_instance_t* src);
void crfsuite_instance_swap(crfsuite_instance_t* x, crfsuite_instance_t* y);
int  crfsuite_instance_append(crfsuite_instance_t* seq, const crfsuite_item_t* item, int label);
int  crfsuite_instance_empty(crfsuite_instance_t* seq);

void crfsuite_data_init(crfsuite_data_t* data);
void crfsuite_data_init_n(crfsuite_data_t* data, int n);
void crfsuite_data_finish(crfsuite_data_t* data);
void crfsuite_data_copy(crfsuite_data_t* dst, const crfsuite_data_t* src);
void crfsuite_data_swap(crfsuite_data_t* x, crfsuite_data_t* y);
int  crfsuite_data_append(crfsuite_data_t* data, const crfsuite_instance_t* inst);
int  crfsuite_data_maxlength(crfsuite_data_t* data);
int  crfsuite_data_totalitems(crfsuite_data_t* data);

void crfsuite_evaluation_init(crfsuite_evaluation_t* eval, int n);
void crfsuite_evaluation_finish(crfsuite_evaluation_t* eval);
void crfsuite_evaluation_clear(crfsuite_evaluation_t* eval);
int crfsuite_evaluation_accmulate(crfsuite_evaluation_t* eval, const crfsuite_instance_t* reference, const int* target);
void crfsuite_evaluation_compute(crfsuite_evaluation_t* eval);
void crfsuite_evaluation_output(crfsuite_evaluation_t* eval, crfsuite_dictionary_t* labels, crfsuite_logging_callback cbm, void *instance);


int crfsuite_interlocked_increment(int *count);
int crfsuite_interlocked_decrement(int *count);


#ifdef    __cplusplus
}
#endif/*__cplusplus*/

#endif/*__CRFSUITE_H__*/
