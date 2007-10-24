#include <os.h>

#include <stdio.h>
#include <stdlib.h>

#if 0
#include <crf.h>
#include "iwa.h"



void read_data(FILE *fpi, crf_data_t* data, crf_dictionary_t* attrs, crf_dictionary_t* labels)
{
	int lid = -1;
	crf_instance_t inst;
	crf_item_t item;
	crf_content_t cont;
	iwa_t* iwa = NULL;
	const iwa_token_t* token = NULL;

	crf_instance_init(&inst);

	iwa = iwa_reader(fpi);
	while (token = iwa_read(iwa), token != NULL) {
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
}

void read_data2(FILE *fpi, crf_data_t* data, crf_dictionary_t* attrs, crf_dictionary_t* labels)
{
	int lid = -1;
	crf_instance_t inst;
	crf_item_t item;
	crf_content_t cont;
	iwa_t* iwa = NULL;
	const iwa_token_t* token = NULL;
	int skip = 0;

	crf_instance_init(&inst);

	iwa = iwa_reader(fpi);
	while (token = iwa_read(iwa), token != NULL) {
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
				lid = labels->to_id(labels, token->attr);
				if (lid == -1) {
					skip = 1;
				}
			} else {
				crf_content_init(&cont);
				cont.aid = attrs->to_id(attrs, token->attr);
				if (0 <= cont.aid) {
					crf_item_append_content(&item, &cont);
				}
			}
			break;
		case IWA_NONE:
			/* Put the training instance. */
			if (!skip) {
				crf_data_append(data, &inst);
			}
			skip = 0;
			crf_instance_finish(&inst);
			break;
		case IWA_COMMENT:
			break;
		}
	}
}

int main(int argc, char *argv[])
{
	FILE *fp = NULL, *fpo = stdout;
	crf_dictionary_t* attrs = NULL;
	crf_dictionary_t* labels = NULL;
	crf_trainer_t* trainer = NULL;
	int first_column = 0;
	evaluation_data_t ed;

	crf_data_t data, eval;

	crf_data_init(&data);
	crf_data_init(&eval);

	crf_create_instance("dictionary", (void**)&attrs);
	crf_create_instance("dictionary", (void**)&labels);
	crf_create_instance("crf1m.trainer", (void**)&trainer);

	fp = fopen(argv[1], "r");
	read_data(fp, &data, attrs, labels);
	fclose(fp);

	fp = fopen(argv[2], "r");
	read_data2(fp, &eval, attrs, labels);
	fclose(fp);

	data.num_labels = labels->num(labels);
	data.num_attrs = attrs->num(attrs);
	data.max_item_length = MAX(crf_data_maxlength(&data), crf_data_maxlength(&eval));

	ed.tbl.tbl = (crf_label_evaluation_t*)calloc(data.num_labels, sizeof(crf_label_evaluation_t));
	ed.tbl.num_labels = labels->num(labels);
	ed.attrs = attrs;
	ed.labels = labels;
	ed.out.num_labels = crf_data_maxlength(&eval);
	ed.out.labels = (int*)calloc(ed.out.num_labels, sizeof(int));
	ed.out.probability = 0;
	ed.eval = &eval;

	fprintf(fpo, "Linear-chain CRF training\n");
	fprintf(fpo, "\n");
	fprintf(fpo, "Number of instances: %d\n", data.num_instances);
	fprintf(fpo, "Number of attributes: %d\n", data.num_attrs);
	fprintf(fpo, "Number of labels: %d\n", data.num_labels);
	fprintf(fpo, "\n");

	trainer->set_message_callback(trainer, NULL, trainer_callback);
	trainer->set_evaluate_callback(trainer, &ed, evaluate_callback);
	trainer->trainer(trainer, &data);

	return 0;
}

#else

#define	APPLICATION_S	"libCRF"
#define	VERSION_S		"0.1"
#define	COPYRIGHT_S		"Copyright (c) 2007 Naoaki Okazaki"

int learn(int argc, char *argv[]);
int main_tag(int argc, char *argv[]);

int main(int argc, char *argv[])
{
	FILE *fpo = stdout;

	fprintf(fpo, APPLICATION_S " " VERSION_S "  " COPYRIGHT_S "\n");
	fprintf(fpo, "\n");

	//return learn(argc-1, argv+1);
	return main_tag(argc-1, argv+1);
}

#endif
