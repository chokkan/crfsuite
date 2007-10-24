#include <os.h>

#include <stdio.h>
#include <stdlib.h>
#include "option.h"

#define	APPLICATION_S	"libCRF"
#define	VERSION_S		"0.1"
#define	COPYRIGHT_S		"Copyright (c) 2007 Naoaki Okazaki"



int main_learn(int argc, char *argv[], const char *argv0);
int main_tag(int argc, char *argv[], const char *argv0);
int main_dump(int argc, char *argv[], const char *argv0);



typedef struct {
	int help;			/**< Show help message and exit. */

	FILE *fpi;
	FILE *fpo;
	FILE *fpe;
} option_t;

static void option_init(option_t* opt)
{
	memset(opt, 0, sizeof(*opt));
}

static void option_finish(option_t* opt)
{
}

BEGIN_OPTION_MAP(parse_options, option_t)

	ON_OPTION(SHORTOPT('h') || LONGOPT("help"))
		opt->help = 1;

END_OPTION_MAP()

static void show_usage(FILE *fp, const char *argv0)
{
	fprintf(fp, "USAGE: %s <COMMAND> [OPTIONS]\n", argv0);
	fprintf(fp, "    COMMAND     Command name to specify the processing\n");
	fprintf(fp, "    OPTIONS     Arguments for the command (optional; command-specific)\n");
	fprintf(fp, "\n");
	fprintf(fp, "COMMAND:\n");
	fprintf(fp, "    learn       Obtain a model from a training set of instances\n");
	fprintf(fp, "    tag         Assign suitable labels to given instances by using the model\n");
	fprintf(fp, "    dump        Output the model in a plain-text format\n");
	fprintf(fp, "\n");
	fprintf(fp, "For the usage of each command, specify -h option followed by the command.\n");
}


int main(int argc, char *argv[])
{
	option_t opt;
	int arg_used = 0;
	const char *command = NULL;
	const char *argv0 = argv[0];
	FILE *fpi = stdin, *fpo = stdout, *fpe = stderr;

	/* Show the copyright information. */
	fprintf(fpe, APPLICATION_S " " VERSION_S "  " COPYRIGHT_S "\n");
	fprintf(fpe, "\n");

	/* Parse the command-line option. */
	option_init(&opt);
	arg_used = option_parse(++argv, --argc, parse_options, &opt);
	if (arg_used < 0) {
		return 1;
	}

	/* Show the help message if specified. */
	if (opt.help) {
		show_usage(fpo, argv0);
		return 0;
	}

	/* Check whether a command is specified in the command-line. */
	if (argc <= arg_used) {
		fprintf(fpe, "ERROR: No command specified. See help (-h) for the usage.\n");
		return 1;
	}

	/* Execute the command. */
	command = argv[arg_used];
	if (strcmp(command, "learn") == 0) {
		return main_learn(argc-arg_used, argv+arg_used, argv0);
	} else if (strcmp(command, "tag") == 0) {
		return main_tag(argc-arg_used, argv+arg_used, argv0);
	} else if (strcmp(command, "dump") == 0) {
		return main_dump(argc-arg_used, argv+arg_used, argv0);
	} else {
		fprintf(fpe, "ERROR: Unrecognized command (%s) specified.\n", command);	
		return 1;
	}

	return 0;
}
