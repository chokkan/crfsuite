#ifndef	__CRF_H__
#define	__CRF_H__

#ifdef	__cplusplus
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
};


/**
 * Content of an item.
 *	A content consists of an attribute id with its frequency in the item.
 */
typedef struct {
	int		aid;	/**< Attribute id. */
	float_t	scale;	/**< Scale factor (frequency) of the attribute. */
} crf_content_t;

/**
 * An item.
 */
typedef struct {
	int				num_contents;	/**< Number of contents associated with the item. */
	int				max_contents;	/**< Maximum number of contents. */
	crf_content_t*	contents;		/**< Array of the contents. */
} crf_item_t;

/**
 * An instance.
 */
typedef struct {
	int			num_items;	/**< Number of items in the instance. */
	int			max_items;	/**< Maximum number of items (internal use). */
	crf_item_t*	items;		/**< Array of the items. */
	int*		labels;		/**< Array of output labels. */
} crf_instance_t;

/**
 * A data.
 */
typedef struct {
	int				num_instances;		/**< Number of instances. */
	int				max_instances;		/**< Maximum number of instances (internal use). */
	int				num_labels;			/**< Number of distinct labels. */
	int				num_attrs;			/**< Number of distinct attributes. */
	int				max_item_length;	/**< Maximum item length. */
	crf_instance_t*	instances;			/**< Array of instances. */
} crf_data_t;

/**
 * An output label sequence.
 */
typedef struct {
	int		num_labels;			/**< Number of output labels. */
	int*	labels;				/**< Array of the output labels. */
	float_t	probability;		/**< Probability of the output labels. */
} crf_output_t;

typedef struct {
	int		num_correct;
	int		num_observation;
	int		num_model;
	int		num_total;
} crf_label_evaluation_t;

typedef struct {
	int num_labels;
	crf_label_evaluation_t* tbl;
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
	int (*trainer)(crf_trainer_t* trainer, crf_data_t* data);
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
	int (*addref)(crf_tagger_t* trainer);

	/**
	 * Decrement the reference counter.
	 */
	int (*release)(crf_tagger_t* trainer);

	/**
	 * Tag an input sequence.
	 */
	int (*tag)(crf_tagger_t* tagger, crf_instance_t *inst, crf_output_t* output);

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
	int (*to_string)(crf_dictionary_t* dic, int id, char **pstr);
	int (*num)(crf_dictionary_t* dic);
	void (*free)(crf_dictionary_t* dic, char *str);
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
	int (*set_float)(crf_params_t* params, const char *name, float_t value);
	int (*set_string)(crf_params_t* params, const char *name, const char *value);

	int (*get_int)(crf_params_t* params, const char *name, int *value);
	int (*get_float)(crf_params_t* params, const char *name, float_t *value);
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
void crf_content_copy(crf_content_t* dst, const crf_content_t* src);
void crf_content_swap(crf_content_t* x, crf_content_t* y);

void crf_item_init(crf_item_t* item);
void crf_item_init_n(crf_item_t* item, int num_contents);
void crf_item_finish(crf_item_t* item);
void crf_item_copy(crf_item_t* dst, const crf_item_t* src);
void crf_item_swap(crf_item_t* x, crf_item_t* y);
int  crf_item_append_content(crf_item_t* item, const crf_content_t* cont);

void crf_instance_init(crf_instance_t* inst);
void crf_instance_init_n(crf_instance_t* inst, int num_items);
void crf_instance_finish(crf_instance_t* inst);
void crf_instance_copy(crf_instance_t* dst, const crf_instance_t* src);
void crf_instance_swap(crf_instance_t* x, crf_instance_t* y);
int  crf_instance_append(crf_instance_t* inst, const crf_item_t* item, int label);

void crf_data_init(crf_data_t* data);
void crf_data_init_n(crf_data_t* data, int n);
void crf_data_finish(crf_data_t* data);
void crf_data_copy(crf_data_t* dst, const crf_data_t* src);
void crf_data_swap(crf_data_t* x, crf_data_t* y);
int  crf_data_append(crf_data_t* data, const crf_instance_t* inst);
int  crf_data_maxlength(crf_data_t* data);
int  crf_data_totalitems(crf_data_t* data);

void crf_evaluation_init(crf_evaluation_t* tbl);
int crf_evaluation_accmulate(crf_evaluation_t* tbl, const crf_instance_t* reference, const crf_output_t* target);

int crf_interlocked_increment(int *count);
int crf_interlocked_decrement(int *count);


#ifdef	__cplusplus
}
#endif/*__cplusplus*/

#endif/*__CRF_H__*/
