#include "os.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "crf.h"
#include "crf1m.h"

typedef struct {
	char type;
	int src;
} intkey_t;

typedef struct {
	char type;
	int ft;
	int src;
} feature_key_t;

typedef struct {
	int		dst;
	float_t	weight;
} feature_data_t;

struct tag_crf1mm {
	int storing;
//	DEPOT *depot;
};
typedef struct tag_crf1mm crf1mm_t;

enum {
	KT_GLOBAL = 'A',
	KT_NUMATTRS,
	KT_NUMLABELS,
	KT_STR2LID,
	KT_LID2STR,
	KT_STR2AID,
	KT_FEATURE,
};

crf1mm_t* crf1mm_open(const char *filename, int storing)
{
	int mode = 0;
	crf1mm_t *model = NULL;

	model = (crf1mm_t*)calloc(1, sizeof(crf1mm_t));
	if (model == NULL) {
		goto error_exit;
	}

#if 0
	mode = storing ? DP_OWRITER | DP_OCREAT : DP_OREADER;
	model->depot = dpopen(filename, mode, -1);
	if (model->depot == NULL) {
		goto error_exit;
	}

	model->storing = storing;
	return model;
#endif

error_exit:
	free(model);
	return NULL;
}

void crf1mm_close(crf1mm_t* model)
{
	if (model != NULL) {
#if 0
		dpclose(model->depot);
#endif
		free(model);
	}
}

int crf1mm_put_num_attrs(crf1mm_t* model, int num_attrs)
{
	char key = KT_NUMATTRS;
#if 0
	dpput(
		model->depot,
		(const char *)&key, sizeof(key),
		(const char *)&num_attrs, sizeof(num_attrs),
		DP_DOVER
		);
#endif
	return 0;
}

int crf1mm_get_num_attrs(crf1mm_t* model, int* ptr_num_attrs)
{
	int dsize = 0;
	char key = KT_NUMATTRS;
#if 0
	int *data = (int*)dpget(
		model->depot,
		(const char *)&key, sizeof(key),
		0, -1, &dsize
		);
	if (data != NULL) {
		*ptr_num_attrs = *data;
		free(data);
		return 0;
	} else {
		return -1;
	}
#endif
}


int crf1mm_put_label(crf1mm_t* model, int lid, const char *value)
{
	int ret = 0;
	intkey_t ikey;
	size_t skeysize = strlen(value) + 2;
	char *skey = (char*)alloca(skeysize);

#if 0
	skey[0] = KT_STR2LID;
	strcpy(&skey[1], value);

	ikey.type = KT_LID2STR;
	ikey.src = lid;

	dpput(
		model->depot,
		(const char *)&ikey, sizeof(ikey),
		value, strlen(value) + 1,
		DP_DOVER
		);
	dpput(
		model->depot,
		skey, skeysize,
		(const char *)&lid, sizeof(lid),
		DP_DOVER
		);
#endif
	return 0;
}

int crf1mm_to_label(crf1mm_t* model, int lid, char **ptr_value)
{
	intkey_t key;
	int dsize = 0;
	char *data = NULL;

#if 0
	key.type = KT_LID2STR;
	key.src = lid;

	data = dpget(
		model->depot,
		(const char *)&key, sizeof(key),
		0, -1, &dsize
		);
	if (data != NULL) {
		*ptr_value = data;
		return 0;
	} else {
		return -1;
	}

#endif
}

int crf1mm_to_lid(crf1mm_t* model, const char *value, int *ptr_lid)
{
	size_t keysize = strlen(value) + 2;
	char *key = (char*)alloca(keysize);
	int dsize = 0;
	int *data = NULL;

#if 0
	key[0] = KT_STR2LID;
	strcpy(&key[1], value);

	data = (int*)dpget(
		model->depot,
		key, keysize,
		0, -1, &dsize
		);
	if (data != NULL) {
		*ptr_lid = *data;
		return 0;
	} else {
		return -1;
	}

#endif
}

int crf1mm_put_attribute(crf1mm_t* model, int aid, const char *value)
{
	int ret = 0;
	size_t keysize = strlen(value) + 2;
	char *key = (char*)alloca(keysize);

#if 0
	/* Construct a key representing the attribute. */
	key[0] = KT_STR2AID;
	strcpy(&key[1], value);

	/* Insert the key and features. */
	dpput(
		model->depot,
		key, keysize,
		(const char *)&aid, sizeof(aid),
		DP_DOVER
		);
#endif
	return 0;
}

int crf1mm_to_aid(crf1mm_t* model, const char *value, int *ptr_aid)
{
	size_t keysize = strlen(value) + 2;
	char *key = (char*)alloca(keysize);
	int dsize = 0;
	int *data = NULL;

#if 0
	key[0] = KT_STR2AID;
	strcpy(&key[1], value);

	data = (int*)dpget(
		model->depot,
		key, keysize,
		0, -1, &dsize
		);
	if (data != NULL) {
		*ptr_aid = *data;
		return 0;
	} else {
		return -1;
	}
#endif
}

int crf1mm_put_feature(crf1mm_t* model, int type, int src, const crf1mm_feature_t* feature)
{
	feature_key_t key;

#if 0
	key.type = KT_FEATURE;
	key.ft = type;
	key.src = src;

	/* Insert the key and features. */
	dpput(
		model->depot,
		(const char *)&key, sizeof(key),
		(const char *)feature, sizeof(crf1mm_feature_t),
		DP_DCAT
		);
#endif
	return 0;
}

int crf1mm_get_features(crf1mm_t* model, int type, int src, crf1mm_feature_t** ptr_features, int *ptr_n)
{
	feature_key_t key;

#if 0

	/* Clear the output arguments. */
	*ptr_features = NULL;
	*ptr_n = 0;

	/* Construct a key. */
	key.type = KT_FEATURE;
	key.ft = type;
	key.src = src;

	/* Retrieve the features associated with the attribute. */
	if (cdb_find(&model->cdb, &key, sizeof(key)) > 0) {
		unsigned vpos = cdb_datapos(&model->cdb);
		unsigned vsize = cdb_datalen(&model->cdb);
		crf1mm_feature_t *features = (crf1mm_feature_t*)malloc(vsize);
		cdb_read(&model->cdb, features, vsize, vpos);

		*ptr_features = features;
		*ptr_n = vsize / sizeof(crf1mm_feature_t);
		return 0;
	} else {
		return 1;
	}
#endif
	return 0;
}
