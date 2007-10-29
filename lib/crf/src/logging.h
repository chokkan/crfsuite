#ifndef	__LOGGING_H__
#define	__LOGGING_H__

typedef struct {
	void *instance;
	crf_logging_callback func;
	int percent;
} logging_t;

void logging(logging_t* lg, const char *format, ...);
void logging_timestamp(logging_t* lg, const char *format);
void logging_progress_start(logging_t* lg);
void logging_progress(logging_t* lg, int percent);
void logging_progress_end(logging_t* lg);

#endif/*__LOGGING_H__*/
