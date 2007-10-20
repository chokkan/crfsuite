#ifndef	__QUARK_H__
#define	__QUARK_H__

struct tag_quark;
typedef struct tag_quark quark_t;

quark_t* quark_new();
void quark_delete(quark_t* qrk);
int quark_get(quark_t* qrk, const char *str);
int quark_to_id(quark_t* qrk, const char *str);
const char *quark_to_string(quark_t* qrk, int qid);
int quark_num(quark_t* qrk);

#endif/*__QUARK_H__*/
