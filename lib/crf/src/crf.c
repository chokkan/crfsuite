/*
 *      CRFsuite library.
 *
 * Copyright (c) 2007-2010, Naoaki Okazaki
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the names of the authors nor the names of its contributors
 *       may be used to endorse or promote products derived from this
 *       software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* $Id$ */

#include <os.h>

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <crfsuite.h>

int crf1dl_create_instance(const char *iid, void **ptr);
int crf_dictionary_create_instance(const char *interface, void **ptr);
int crf1m_create_instance_from_file(const char *filename, void **ptr);

int crf_create_instance(const char *iid, void **ptr)
{
    int ret = 
        crf1dl_create_instance(iid, ptr) == 0 ||
        crf_dictionary_create_instance(iid, ptr) == 0;

    return ret;
}

int crf_create_instance_from_file(const char *filename, void **ptr)
{
    int ret = crf1m_create_instance_from_file(filename, ptr);
    return ret;
}



void crf_content_init(crf_content_t* cont)
{
    memset(cont, 0, sizeof(*cont));
    cont->scale = 1;
}

void crf_content_set(crf_content_t* cont, int aid, floatval_t scale)
{
    crf_content_init(cont);
    cont->aid = aid;
    cont->scale = scale;
}

void crf_content_copy(crf_content_t* dst, const crf_content_t* src)
{
    dst->aid = src->aid;
    dst->scale = src->scale;
}

void crf_content_swap(crf_content_t* x, crf_content_t* y)
{
    crf_content_t tmp = *x;
    x->aid = y->aid;
    x->scale = y->scale;
    y->aid = tmp.aid;
    y->scale = tmp.scale;
}



void crf_item_init(crf_item_t* item)
{
    memset(item, 0, sizeof(*item));
}

void crf_item_init_n(crf_item_t* item, int num_contents)
{
    crf_item_init(item);
    item->num_contents = num_contents;
    item->max_contents = num_contents;
    item->contents = (crf_content_t*)calloc(num_contents, sizeof(crf_content_t));
}

void crf_item_finish(crf_item_t* item)
{
    free(item->contents);
    crf_item_init(item);
}

void crf_item_copy(crf_item_t* dst, const crf_item_t* src)
{
    int i;

    dst->num_contents = src->num_contents;
    dst->max_contents = src->max_contents;
    dst->label = src->label;
    dst->contents = (crf_content_t*)calloc(dst->num_contents, sizeof(crf_content_t));
    for (i = 0;i < dst->num_contents;++i) {
        crf_content_copy(&dst->contents[i], &src->contents[i]);
    }
}

void crf_item_swap(crf_item_t* x, crf_item_t* y)
{
    crf_item_t tmp = *x;
    x->num_contents = y->num_contents;
    x->max_contents = y->max_contents;
    x->label = y->label;
    x->contents = y->contents;
    y->num_contents = tmp.num_contents;
    y->max_contents = tmp.max_contents;
    y->label = tmp.label;
    y->contents = tmp.contents;
}

int crf_item_append_content(crf_item_t* item, const crf_content_t* cont)
{
    if (item->max_contents <= item->num_contents) {
        item->max_contents = (item->max_contents + 1) * 2;
        item->contents = (crf_content_t*)realloc(
            item->contents, sizeof(crf_content_t) * item->max_contents);
    }
    crf_content_copy(&item->contents[item->num_contents++], cont);
    return 0;
}

int  crf_item_empty(crf_item_t* item)
{
    return (item->num_contents == 0);
}




void crf_sequence_init(crf_sequence_t* inst)
{
    memset(inst, 0, sizeof(*inst));
}

void crf_sequence_init_n(crf_sequence_t* inst, int num_items)
{
    crf_sequence_init(inst);
    inst->num_items = num_items;
    inst->max_items = num_items;
    inst->items = (crf_item_t*)calloc(num_items, sizeof(crf_item_t));
}

void crf_sequence_finish(crf_sequence_t* inst)
{
    int i;

    for (i = 0;i < inst->num_items;++i) {
        crf_item_finish(&inst->items[i]);
    }
    free(inst->items);
    crf_sequence_init(inst);
}

void crf_sequence_copy(crf_sequence_t* dst, const crf_sequence_t* src)
{
    int i;

    dst->num_items = src->num_items;
    dst->max_items = src->max_items;
    dst->items = (crf_item_t*)calloc(dst->num_items, sizeof(crf_item_t));
    for (i = 0;i < dst->num_items;++i) {
        crf_item_copy(&dst->items[i], &src->items[i]);
    }
}

void crf_sequence_swap(crf_sequence_t* x, crf_sequence_t* y)
{
    crf_sequence_t tmp = *x;
    x->num_items = y->num_items;
    x->max_items = y->max_items;
    x->items = y->items;
    y->num_items = tmp.num_items;
    y->max_items = tmp.max_items;
    y->items = tmp.items;
}

int crf_sequence_append(crf_sequence_t* inst, const crf_item_t* item, int label)
{
    if (inst->max_items <= inst->num_items) {
        inst->max_items = (inst->max_items + 1) * 2;
        inst->items = (crf_item_t*)realloc(inst->items, sizeof(crf_item_t) * inst->max_items);
    }
    crf_item_copy(&inst->items[inst->num_items], item);
    inst->items[inst->num_items].label = label;
    ++inst->num_items;
    return 0;
}

int  crf_sequence_empty(crf_sequence_t* inst)
{
    return (inst->num_items == 0);
}




void crf_data_init(crf_data_t* data)
{
    memset(data, 0, sizeof(*data));
}

void crf_data_init_n(crf_data_t* data, int n)
{
    crf_data_init(data);
    data->num_instances = n;
    data->max_instances = n;
    data->instances = (crf_sequence_t*)calloc(n, sizeof(crf_sequence_t));
}

void crf_data_finish(crf_data_t* data)
{
    int i;

    for (i = 0;i < data->num_instances;++i) {
        crf_sequence_finish(&data->instances[i]);
    }
    free(data->instances);
    crf_data_init(data);
}

void crf_data_copy(crf_data_t* dst, const crf_data_t* src)
{
    int i;

    dst->num_instances = src->num_instances;
    dst->max_instances = src->max_instances;
    dst->instances = (crf_sequence_t*)calloc(dst->num_instances, sizeof(crf_sequence_t));
    for (i = 0;i < dst->num_instances;++i) {
        crf_sequence_copy(&dst->instances[i], &src->instances[i]);
    }
}

void crf_data_swap(crf_data_t* x, crf_data_t* y)
{
    crf_data_t tmp = *x;
    x->num_instances = y->num_instances;
    x->max_instances = y->max_instances;
    x->instances = y->instances;
    y->num_instances = tmp.num_instances;
    y->max_instances = tmp.max_instances;
    y->instances = tmp.instances;
}

int  crf_data_append(crf_data_t* data, const crf_sequence_t* inst)
{
    if (0 < inst->num_items) {
        if (data->max_instances <= data->num_instances) {
            data->max_instances = (data->max_instances + 1) * 2;
            data->instances = (crf_sequence_t*)realloc(
                data->instances, sizeof(crf_sequence_t) * data->max_instances);
        }
        crf_sequence_copy(&data->instances[data->num_instances++], inst);
    }
    return 0;
}

int crf_data_maxlength(crf_data_t* data)
{
    int i, T = 0;
    for (i = 0;i < data->num_instances;++i) {
        if (T < data->instances[i].num_items) {
            T = data->instances[i].num_items;
        }
    }
    return T;
}

int  crf_data_totalitems(crf_data_t* data)
{
    int i, n = 0;
    for (i = 0;i < data->num_instances;++i) {
        n += data->instances[i].num_items;
    }
    return n;
}

void crf_output_init(crf_output_t* output)
{
    memset(output, 0, sizeof(*output));
}

void crf_output_init_n(crf_output_t* output, int n)
{
    crf_output_init(output);
    output->labels = (int*)calloc(n, sizeof(int));
    if (output->labels != NULL) {
        output->num_labels = n;
    }
}

void crf_output_finish(crf_output_t* output)
{
    free(output->labels);
    crf_output_init(output);    
}


static char *safe_strncpy(char *dst, const char *src, size_t n)
{
    strncpy(dst, src, n-1);
    dst[n-1] = 0;
    return dst;
}

void crf_evaluation_init(crf_evaluation_t* eval, int n)
{
    memset(eval, 0, sizeof(*eval));
    eval->tbl = (crf_label_evaluation_t*)calloc(n+1, sizeof(crf_label_evaluation_t));
    if (eval->tbl != NULL) {
        eval->num_labels = n;
    }
}

void crf_evaluation_clear(crf_evaluation_t* eval)
{
    int i;
    for (i = 0;i <= eval->num_labels;++i) {
        memset(&eval->tbl[i], 0, sizeof(eval->tbl[i]));
    }

    eval->item_total_correct = 0;
    eval->item_total_num = 0;
    eval->item_total_model = 0;
    eval->item_total_observation = 0;
    eval->item_accuracy = 0;

    eval->inst_total_correct = 0;
    eval->inst_total_num = 0;
    eval->inst_accuracy = 0;

    eval->macro_precision = 0;
    eval->macro_recall = 0;
    eval->macro_fmeasure = 0;
}

void crf_evaluation_finish(crf_evaluation_t* eval)
{
    free(eval->tbl);
    memset(eval, 0, sizeof(*eval));
}

int crf_evaluation_accmulate(crf_evaluation_t* eval, const crf_sequence_t* reference, const crf_output_t* target)
{
    int t, nc = 0;

    /* Make sure that the reference and target sequences have the output labels of the same length. */
    if (reference->num_items != target->num_labels) {
        return 1;
    }

    for (t = 0;t < target->num_labels;++t) {
        int lr = reference->items[t].label;
        int lt = target->labels[t];

        if (eval->num_labels <= lr || eval->num_labels <= lt) {
            return 1;
        }

        ++eval->tbl[lr].num_observation;
        ++eval->tbl[lt].num_model;
        if (lr == lt) {
            ++eval->tbl[lr].num_correct;
            ++nc;
        }
        ++eval->item_total_num;
    }

    if (nc == target->num_labels) {
        ++eval->inst_total_correct;
    }
    ++eval->inst_total_num;

    return 0;
}

void crf_evaluation_compute(crf_evaluation_t* eval)
{
    int i;

    for (i = 0;i <= eval->num_labels;++i) {
        crf_label_evaluation_t* lev = &eval->tbl[i];

        /* Do not evaluate labels that does not in the test data. */
        if (lev->num_observation == 0) {
            continue;
        }

        /* Sum the number of correct labels for accuracy calculation. */
        eval->item_total_correct += lev->num_correct;
        eval->item_total_model += lev->num_model;
        eval->item_total_observation += lev->num_observation;

        /* Initialize the precision, recall, and f1-measure values. */
        lev->precision = 0;
        lev->recall = 0;
        lev->fmeasure = 0;

        /* Compute the precision, recall, and f1-measure values. */
        if (lev->num_model > 0) {
            lev->precision = lev->num_correct / (double)lev->num_model;
        }
        if (lev->num_observation > 0) {
            lev->recall = lev->num_correct / (double)lev->num_observation;
        }
        if (lev->precision + lev->recall > 0) {
            lev->fmeasure = lev->precision * lev->recall * 2 / (lev->precision + lev->recall);
        }

        /* Exclude unknown labels from calculation of macro-average values. */
        if (i != eval->num_labels) {
            eval->macro_precision += lev->precision;
            eval->macro_recall += lev->recall;
            eval->macro_fmeasure += lev->fmeasure;
        }
    }

    /* Copute the macro precision, recall, and f1-measure values. */
    eval->macro_precision /= eval->num_labels;
    eval->macro_recall /= eval->num_labels;
    eval->macro_fmeasure /= eval->num_labels;

    /* Compute the item accuracy. */
    eval->item_accuracy = 0;
    if (0 < eval->item_total_num) {
        eval->item_accuracy = eval->item_total_correct / (double)eval->item_total_num;
    }

    /* Compute the instance accuracy. */
    eval->inst_accuracy = 0;
    if (0 < eval->inst_total_num) {
        eval->inst_accuracy = eval->inst_total_correct / (double)eval->inst_total_num;
    }
}

void crf_evaluation_output(crf_evaluation_t* eval, crf_dictionary_t* labels, FILE *fpo)
{
    int i;
    const char *lstr = NULL;

    fprintf(fpo, "Performance by label (#match, #model, #ref) (precision, recall, F1):\n");

    for (i = 0;i < eval->num_labels;++i) {
        const crf_label_evaluation_t* lev = &eval->tbl[i];

        labels->to_string(labels, i, &lstr);
        if (lstr == NULL) lstr = "[UNKNOWN]";

        if (lev->num_observation == 0) {
            fprintf(fpo, "    %s: (%d, %d, %d) (******, ******, ******)\n",
                lstr, lev->num_correct, lev->num_model, lev->num_observation
                );
        } else {
            fprintf(fpo, "    %s: (%d, %d, %d) (%1.4f, %1.4f, %1.4f)\n",
                lstr, lev->num_correct, lev->num_model, lev->num_observation,
                lev->precision, lev->recall, lev->fmeasure
                );
        }
        labels->free(labels, lstr);
    }
    fprintf(fpo, "Macro-average precision, recall, F1: (%f, %f, %f)\n",
        eval->macro_precision, eval->macro_recall, eval->macro_fmeasure
        );
    fprintf(fpo, "Item accuracy: %d / %d (%1.4f)\n",
        eval->item_total_correct, eval->item_total_num, eval->item_accuracy
        );
    fprintf(fpo, "Instance accuracy: %d / %d (%1.4f)\n",
        eval->inst_total_correct, eval->inst_total_num, eval->inst_accuracy
        );
}

int crf_interlocked_increment(int *count)
{
    return ++(*count);
}

int crf_interlocked_decrement(int *count)
{
    return --(*count);
}
