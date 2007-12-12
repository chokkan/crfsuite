/*
 *		Learn command for CRFsuite frontend.
 *
 *	Copyright (C) 2007 Naoaki Okazaki
 *
 *	This program is free software: you can redistribute it and/or modify
 *	it under the terms of the GNU General Public License as published by
 *	the Free Software Foundation, either version 3 of the License, or
 *	any later version.
 *
 *	This program is distributed in the hope that it will be useful,
 *	but WITHOUT ANY WARRANTY; without even the implied warranty of
 *	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *	GNU General Public License for more details.
 *
 *	You should have received a copy of the GNU General Public License
 *	along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/* $Id$ */

#include <os.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <crf.h>
#include "option.h"
#include "readdata.h"

#define	SAFE_RELEASE(obj)	if ((obj) != NULL) { (obj)->release(obj); (obj) = NULL; }
#define	MAX(a, b)	((a) < (b) ? (b) : (a))


typedef struct {
	char *model;
	char *training;
	char *evaluation;

	int help;

	int num_params;
	char **params;
} learn_option_t;

static void learn_option_init(learn_option_t* opt)
{
	memset(opt, 0, sizeof(*opt));
	opt->num_params = 0;
	opt->model = strdup("crfsuite.model");
}

static void learn_option_finish(learn_option_t* opt)
{
	int i;

	free(opt->model);
	free(opt->training);
	free(opt->evaluation);

	for (i = 0;i < opt->num_params;++i) {
		free(opt->params[i]);
	}
	free(opt->params);
}

BEGIN_OPTION_MAP(parse_learn_options, learn_option_t)

	ON_OPTION_WITH_ARG(SHORTOPT('m') || LONGOPT("model"))
		free(opt->model);
		opt->model = strdup(arg);

	ON_OPTION_WITH_ARG(SHORTOPT('t') || LONGOPT("test"))
		free(opt->evaluation);
		opt->evaluation = strdup(arg);

	ON_OPTION(SHORTOPT('h') || LONGOPT("help"))
		opt->help = 1;

	ON_OPTION_WITH_ARG(SHORTOPT('p') || LONGOPT("param"))
		opt->params = (char **)realloc(opt->params, sizeof(char*) * (opt->num_params + 1));
		opt->params[opt->num_params] = strdup(arg);
		++opt->num_params;

END_OPTION_MAP()

static void show_usage(FILE *fp, const char *argv0, const char *command)
{
	fprintf(fp, "USAGE: %s %s [OPTIONS] [DATA]\n", argv0, command);
	fprintf(fp, "Obtain a model from a training set of instances given by a file (DATA).\n");
	fprintf(fp, "If argument DATA is omitted or '-', this utility reads a data from STDIN.\n");
	fprintf(fp, "\n");
	fprintf(fp, "OPTIONS:\n");
	fprintf(fp, "    -m, --model=MODEL   Store the obtained model in a file (MODEL)\n");
	fprintf(fp, "    -t, --test=TEST     Report the performance of the model on a data (TEST)\n");
	fprintf(fp, "    -h, --help          Show the usage of this command and exit\n");
}



typedef struct {
	FILE *fpo;
	crf_data_t* data;
	crf_evaluation_t* eval;
	crf_dictionary_t* attrs;
	crf_dictionary_t* labels;
} callback_data_t;

static int message_callback(void *instance, const char *format, va_list args)
{
	callback_data_t* cd = (callback_data_t*)instance;
	FILE *fpo = cd->fpo;
	vfprintf(fpo, format, args);
	fflush(fpo);
	return 0;
}

static int evaluate_callback(void *instance, crf_tagger_t* tagger)
{
	int i, ret = 0;
	crf_output_t output;
	callback_data_t* cd = (callback_data_t*)instance;
	FILE *fpo = cd->fpo;
	crf_data_t* data = cd->data;
	crf_dictionary_t* labels = cd->labels;

	/* Do nothing if no test data was given. */
	if (data->num_instances == 0) {
		return 0;
	}

	/* Clear the evaluation table. */
	crf_evaluation_clear(cd->eval);

	/* Tag the evaluation instances and accumulate the performance. */
	for (i = 0;i < data->num_instances;++i) {
		/* An instance to be tagged. */
		crf_instance_t* instance = &data->instances[i];

		crf_output_init(&output);

		/* Tag an instance (ignoring any error occurrence). */
		ret = tagger->tag(tagger, instance, &output);

		/* Accumulate the tagging performance. */
		crf_evaluation_accmulate(cd->eval, instance, &output);
	}

	/* Compute the performance. */
	crf_evaluation_compute(cd->eval);

	/* Report the performance. */
	crf_evaluation_output(cd->eval, labels, fpo);

	return 0;
}

int main_learn(int argc, char *argv[], const char *argv0)
{
	int i, ret = 0, arg_used = 0;
	time_t ts;
	char timestamp[80];
	clock_t clk_begin, clk_current;
	learn_option_t opt;
	const char *command = argv[0];
	FILE *fp = NULL, *fpi = stdin, *fpo = stdout, *fpe = stderr;
	callback_data_t cd;
	crf_data_t data_train, data_test;
	crf_evaluation_t eval;
	crf_trainer_t *trainer = NULL;
	crf_dictionary_t *attrs = NULL, *labels = NULL;

	/* Initializations. */
	learn_option_init(&opt);
	crf_data_init(&data_train);
	crf_data_init(&data_test);
	crf_evaluation_init(&eval, 0);

	/* Parse the command-line option. */
	arg_used = option_parse(++argv, --argc, parse_learn_options, &opt);
	if (arg_used < 0) {
		ret = 1;
		goto force_exit;
	}

	/* Show the help message for this command if specified. */
	if (opt.help) {
		show_usage(fpo, argv0, command);
		goto force_exit;
	}

	/* Set a training file. */
	if (arg_used < argc) {
		opt.training = strdup(argv[arg_used]);
	} else {
		opt.training = strdup("-");	/* STDIN. */
	}

	/* Create dictionaries for attributes and labels. */
	ret = crf_create_instance("dictionary", (void**)&attrs);
	if (!ret) {
		fprintf(fpe, "ERROR: Failed to create a dictionary instance.\n");
		ret = 1;
		goto force_exit;
	}
	ret = crf_create_instance("dictionary", (void**)&labels);
	if (!ret) {
		fprintf(fpe, "ERROR: Failed to create a dictionary instance.\n");
		ret = 1;
		goto force_exit;
	}

	/* Create a trainer instance. */
	ret = crf_create_instance("trainer.crf1m", (void**)&trainer);
	if (!ret) {
		fprintf(fpe, "ERROR: Failed to create a trainer instance.\n");
		ret = 1;
		goto force_exit;
	}

	/* Set parameters. */
	for (i = 0;i < opt.num_params;++i) {
		char *value = NULL;
		char *name = opt.params[i];
		crf_params_t* params = trainer->params(trainer);
		
		/* Split the parameter argument by the first '=' character. */
		value = strchr(name, '=');
		if (value != NULL) {
			*value++ = 0;
		}

		params->set(params, name, value);
		params->release(params);
	}

	/* Open the training data. */
	fp = (strcmp(opt.training, "-") == 0) ? fpi : fopen(opt.training, "r");
	if (fp == NULL) {
		fprintf(fpe, "ERROR: Failed to open the training data.\n");
		ret = 1;
		goto force_exit;		
	}

	/* Log the start time. */
	time(&ts);
	strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%SZ", gmtime(&ts));
	fprintf(fpo, "Start time of the training: %s\n", timestamp);
	fprintf(fpo, "\n");

	/* Read the training data. */
	fprintf(fpo, "Reading the training data\n");
	clk_begin = clock();
	read_data(fp, fpo, &data_train, attrs, labels);
	clk_current = clock();
	if (fp != fpi) fclose(fp);

	/* Report the statistics of the training data. */
	fprintf(fpo, "Number of instances: %d\n", data_train.num_instances);
	fprintf(fpo, "Total number of items: %d\n", crf_data_totalitems(&data_train));
	fprintf(fpo, "Number of attributes: %d\n", labels->num(attrs));
	fprintf(fpo, "Number of labels: %d\n", labels->num(labels));
	fprintf(fpo, "Seconds required: %.3f\n", (clk_current - clk_begin) / (double)CLOCKS_PER_SEC);
	fprintf(fpo, "\n");
	fflush(fpo);

	/* Read a test data if necessary */
	if (opt.evaluation != NULL) {
		fp = fopen(opt.evaluation, "r");
		if (fp == NULL) {
			fprintf(fpe, "ERROR: Failed to open the evaluation data.\n");
			ret = 1;
			goto force_exit;		
		}

		/* Read the test data. */
		fprintf(fpo, "Reading the evaluation data\n");
		clk_begin = clock();
		read_data(fp, fpo, &data_test, attrs, labels);
		clk_current = clock();
		fclose(fp);

		/* Report the statistics of the test data. */
		fprintf(fpo, "Number of instances: %d\n", data_test.num_instances);
		fprintf(fpo, "Number of total items: %d\n", crf_data_totalitems(&data_test));
		fprintf(fpo, "Seconds required: %.3f\n", (clk_current - clk_begin) / (double)CLOCKS_PER_SEC);
		fprintf(fpo, "\n");
		fflush(fpo);
	}

	/* Fill the supplementary information for the data. */
	data_train.num_labels = labels->num(labels);
	data_train.num_attrs = labels->num(attrs);
	data_train.max_item_length = crf_data_maxlength(&data_train);

	/* Initialize an evaluation object. */
	crf_evaluation_finish(&eval);
	crf_evaluation_init(&eval, labels->num(labels));

	/* Fill the callback data. */
	cd.fpo = fpo;
	cd.eval = &eval;
	cd.attrs = attrs;
	cd.labels = labels;
	cd.data = &data_test;

	/* Set callback procedures that receive messages and taggers. */
	trainer->set_message_callback(trainer, &cd, message_callback);
	trainer->set_evaluate_callback(trainer, &cd, evaluate_callback);

	/* Start training. */
	if (ret = trainer->trainer(trainer, &data_train)) {
		goto force_exit;
	}

	/* Write out the obtained model. */
	if (opt.model != NULL) {
		if (ret = trainer->save(trainer, opt.model, attrs, labels)) {
			goto force_exit;
		}
	}

	/* Log the end time. */
	time(&ts);
	strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%SZ", gmtime(&ts));
	fprintf(fpo, "End time of the training: %s\n", timestamp);
	fprintf(fpo, "\n");

force_exit:
	SAFE_RELEASE(trainer);
	SAFE_RELEASE(labels);
	SAFE_RELEASE(attrs);

	crf_data_finish(&data_test);
	crf_data_finish(&data_train);
	crf_evaluation_finish(&eval);
	learn_option_finish(&opt);

	return ret;
}
