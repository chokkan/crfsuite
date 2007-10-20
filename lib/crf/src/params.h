#ifndef	__PARAMS_H__
#define	__PARAMS_H__

crf_params_t* params_create_instance();
int params_add_int(crf_params_t* params, const char *name, int value);
int params_add_float(crf_params_t* params, const char *name, float_t value);
int params_add_string(crf_params_t* params, const char *name, const char *value);

#define	BEGIN_PARAM_MAP(params, mode) \
	do { \
		int __ret = 0; \
		int __mode = mode; \
		crf_params_t* __params = params;

#define	END_PARAM_MAP() \
	} while (0) ;

#define	DDX_PARAM_INT(name, var, defval) \
	if (__mode < 0) \
		__ret = __params->get_int(__params, name, &var); \
	else if (__mode > 0) \
		__ret = __params->set_int(__params, name, var); \
	else \
		__ret = params_add_int(__params, name, defval);

#define	DDX_PARAM_FLOAT(name, var, defval) \
	if (__mode < 0) \
		__ret = __params->get_float(__params, name, &var); \
	else if (__mode > 0) \
		__ret = __params->set_float(__params, name, var); \
	else \
		__ret = params_add_float(__params, name, defval);

#define	DDX_PARAM_STRING(name, var, defval) \
	if (__mode < 0) \
		__ret = __params->get_string(__params, name, &var); \
	else if (__mode > 0) \
		__ret = __params->set_string(__params, name, var); \
	else \
		__ret = params_add_string(__params, name, defval);

#endif/*__PARAMS_H__*/
