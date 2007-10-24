#include <os.h>

#include <stdio.h>
#include <stdlib.h>

#include <crf.h>
#include "iwa.h"

static int progress(FILE *fpo, int prev, int current)
{
	while (prev < current) {
		++prev;
		if (prev % 2 == 0) {
			if (prev % 10 == 0) {
				fprintf(fpo, "%d", prev / 10);
			} else {
				fprintf(fpo, ".", prev / 10);
			}
		}
	}
	return prev;
}

void read_data(FILE *fpi, FILE *fpo, crf_data_t* data, crf_dictionary_t* attrs, crf_dictionary_t* labels)
{
	int lid = -1;
	crf_instance_t inst;
	crf_item_t item;
	crf_content_t cont;
	iwa_t* iwa = NULL;
	const iwa_token_t* token = NULL;
	long filesize = 0, begin = 0, offset = 0;
	int i, prev = 0, current = 0;

	/* Initialize the instance.*/
	crf_instance_init(&inst);

	/* Obtain the file size. */
	begin = ftell(fpi);
	fseek(fpi, 0, SEEK_END);
	filesize = ftell(fpi) - begin;
	fseek(fpi, begin, SEEK_SET);

	/* */
	fprintf(fpo, "0");
	prev = 0;

	iwa = iwa_reader(fpi);
	while (token = iwa_read(iwa), token != NULL) {
		/* Progress report. */
		offset = ftell(fpi);
		current = (int)((offset - begin) * 100.0 / (double)filesize);
		prev = progress(fpo, prev, current);

		switch (token->type) {
		case IWA_BOI:
			/* Initialize an item. */
			lid = -1;
			crf_item_init(&item);
			break;
		case IWA_EOI:
			/* Append the item to the instance. */
			crf_instance_append(&inst, &item, lid);
			crf_item_finish(&item);
			break;
		case IWA_ITEM:
			if (lid == -1) {
				lid = labels->get(labels, token->attr);
			} else {
				crf_content_init(&cont);
				cont.aid = attrs->get(attrs, token->attr);
				crf_item_append_content(&item, &cont);
			}
			break;
		case IWA_NONE:
			/* Put the training instance. */
			crf_data_append(data, &inst);
			crf_instance_finish(&inst);
			break;
		case IWA_COMMENT:
			break;
		}
	}

	progress(fpo, prev, 100);
	fprintf(fpo, "\n");
}
