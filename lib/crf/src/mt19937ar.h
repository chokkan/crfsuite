#ifndef __MT19937AR_H__
#define __MT19937AR_H__

#ifdef __cplusplus
extern "C" {
#endif

void mt_init_genrand(unsigned long s);
void mt_init_by_array(unsigned long init_key[], int key_length);
unsigned long mt_genrand_int32(void);
long mt_genrand_int31(void);
double mt_genrand_real1(void);
double mt_genrand_real2(void);
double mt_genrand_real3(void);
double mt_genrand_res53(void);

#ifdef __cplusplus
};
#endif

#endif/*__MT19937AR_H__*/
