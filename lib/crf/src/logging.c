#include <os.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <crf.h>
#include "logging.h"

void logging(logging_t* lg, const char *format, ...)
{
	va_list args;
	va_start(args, format);

	if (lg->func != NULL) {
		lg->func(lg->instance, format, args);
	}
}

void logging_timestamp(logging_t* lg, const char *format)
{
	time_t ts;
	char timestamp[80];

	time(&ts);
	strftime(
		timestamp, sizeof(timestamp),
		"%Y-%m-%dT%H:%M:%SZ",
		gmtime(&ts)
		);
	logging(lg, format, timestamp);
}

void logging_progress_start(logging_t* lg)
{
	lg->percent = 0;
	logging(lg, "0");
}

void logging_progress(logging_t* lg, int percent)
{
	while (lg->percent < percent) {
		++lg->percent;
		if (lg->percent % 2 == 0) {
			if (lg->percent % 10 == 0) {
				logging(lg, "%d", lg->percent / 10);
			} else {
				logging(lg, ".");
			}
		}
	}
}

void logging_progress_end(logging_t* lg)
{
	logging_progress(lg, 100);
	logging(lg, "\n");
}
