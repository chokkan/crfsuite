/*
 *		A parser for Item With Attributes (IWA) format.
 *
 *		Copyright (c) 2007 Naoaki Okazaki
 *
 * This software is provided 'as-is', without any express or implied
 * warranty.  In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions (known as zlib license):
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgment in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 *
 * Naoaki Okazaki <okazaki at chokkan dot org>
 *
 */

/* $Id:$ */

#ifndef	__IWA_H__
#define	__IWA_H__

#ifdef	__cplusplus
extern "C" {
#endif/*__cplusplus*/

typedef struct tag_iwa iwa_t;

enum {
	IWA_NONE,
	IWA_BOI,
	IWA_EOI,
	IWA_ITEM,
	IWA_COMMENT,
};

struct tag_iwa_token {
	int type;
	const char *attr;
	const char *value;
	const char *comment;
};
typedef struct tag_iwa_token iwa_token_t;

iwa_t* iwa_reader(FILE *fp);
const iwa_token_t* iwa_read(iwa_t* iwa);
void iwa_delete(iwa_t* iwa);

#ifdef	__cplusplus
}
#endif/*__cplusplus*/

#endif/*__IWA_H__*/
