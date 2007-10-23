#include <os.h>

#include <stdlib.h>
#include <string.h>

#include <crf.h>
#include "quark.h"

static int dictionary_addref(crf_dictionary_t* dic)
{
	return crf_interlocked_increment(&dic->nref);
}

static int dictionary_release(crf_dictionary_t* dic)
{
	int count = crf_interlocked_decrement(&dic->nref);
	if (count == 0) {
		quark_t *qrk = (quark_t*)dic->internal;
		quark_delete(qrk);
		free(dic);
	}
	return count;
}

static int dictionary_get(crf_dictionary_t* dic, const char *str)
{
	quark_t *qrk = (quark_t*)dic->internal;
	return quark_get(qrk, str);
}

static int dictionary_to_id(crf_dictionary_t* dic, const char *str)
{
	quark_t *qrk = (quark_t*)dic->internal;
	return quark_to_id(qrk, str);	
}

static int dictionary_to_string(crf_dictionary_t* dic, int id, char **pstr)
{
	quark_t *qrk = (quark_t*)dic->internal;
	const char *str = quark_to_string(qrk, id);
	if (str != NULL) {
		*pstr = strdup(str);
		return 0;
	} else {
		return 1;
	}
}

static int dictionary_num(crf_dictionary_t* dic)
{
	quark_t *qrk = (quark_t*)dic->internal;
	return quark_num(qrk);
}

static void dictionary_free(crf_dictionary_t* dic, char *str)
{
	free(str);
}

int crf_dictionary_create_instance(const char *interface, void **ptr)
{
	if (strcmp(interface, "dictionary") == 0) {
		crf_dictionary_t* dic = (crf_dictionary_t*)calloc(1, sizeof(crf_dictionary_t));

		if (dic != NULL) {
			dic->internal = quark_new();
			dic->nref = 1;
			dic->addref = dictionary_addref;
			dic->release = dictionary_release;
			dic->get = dictionary_get;
			dic->to_id = dictionary_to_id;
			dic->to_string = dictionary_to_string;
			dic->num = dictionary_num;
			dic->free = dictionary_free;
			*ptr = dic;
			return 0;
		} else {
			return -1;
		}
	} else {
		return 1;
	}
}
