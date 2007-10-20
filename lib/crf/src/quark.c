#include "os.h"
#include <stdlib.h>
#include <string.h>
#include "rumavl.h"
#include "quark.h"

typedef struct {
	char *str;
	int qid;
} record_t;

struct tag_quark {
	int num;
	int max;
	RUMAVL* string_to_id;
	char **id_to_string;
};

static int keycmp(const void *_x, const void *_y, size_t n)
{
	const record_t* x = (const record_t*)_x;
	const record_t* y = (const record_t*)_y;
	return strcmp(x->str, y->str);
}

static int owcb(RUMAVL *tree, void *_x, const void *_y, void *udata)
{
	record_t* x = (record_t*)_x;
	free(x->str);
	return 0;
}

static int delcb(RUMAVL *tree, void *_record, void *udata)
{
	record_t* record = (record_t*)_record;
	free(record->str);
	return 0;
}

quark_t* quark_new()
{
	quark_t* qrk = (quark_t*)malloc(sizeof(quark_t));
	if (qrk != NULL) {
		qrk->num = 0;
		qrk->max = 0;
		qrk->string_to_id = rumavl_new(sizeof(record_t), keycmp);
		if (qrk->string_to_id != NULL) {
			rumavl_cb(qrk->string_to_id)->del = delcb;
			rumavl_cb(qrk->string_to_id)->ow = owcb;
		}
		qrk->id_to_string = NULL;
	}
	return qrk;
}

void quark_delete(quark_t* qrk)
{
	rumavl_destroy(qrk->string_to_id);
}

int quark_get(quark_t* qrk, const char *str)
{
	record_t key, *record = NULL;

	key.str = (char *)str;
	record = (record_t*)rumavl_find(qrk->string_to_id, &key);
	if (record == NULL) {
		char *newstr = strdup(str);

		if (qrk->max <= qrk->num) {
			qrk->max = (qrk->max + 1) * 2;
			qrk->id_to_string = (char **)realloc(qrk->id_to_string, sizeof(char *) * qrk->max);
		}

		qrk->id_to_string[qrk->num] = newstr;
		key.str = newstr;
		key.qid = qrk->num;
		rumavl_insert(qrk->string_to_id, &key);

		++qrk->num;
		return key.qid;
	} else {
		return record->qid;
	}	
}

int quark_to_id(quark_t* qrk, const char *str)
{
	record_t key, *record = NULL;

	key.str = (char *)str;
	record = (record_t*)rumavl_find(qrk->string_to_id, &key);
	return (record != NULL) ? record->qid : -1;
}

const char *quark_to_string(quark_t* qrk, int qid)
{
	return (qid < qrk->num) ? qrk->id_to_string[qid] : NULL;
}

int quark_num(quark_t* qrk)
{
	return qrk->num;
}


#if 0
int main(int argc, char *argv[])
{
	quark_t *qrk = quark_new();
	int qid = 0;

	qid = quark_get(qrk, "zero");
	qid = quark_get(qrk, "one");
	qid = quark_get(qrk, "zero");
	qid = quark_to_id(qrk, "three");
	qid = quark_get(qrk, "two");
	qid = quark_get(qrk, "three");
	qid = quark_to_id(qrk, "three");
	qid = quark_get(qrk, "zero");
	qid = quark_get(qrk, "one");

	printf("%s\n", quark_to_string(qrk, 0));
	printf("%s\n", quark_to_string(qrk, 1));
	printf("%s\n", quark_to_string(qrk, 2));
	printf("%s\n", quark_to_string(qrk, 3));

	quark_delete(qrk);
	
	return 0;
}
#endif
