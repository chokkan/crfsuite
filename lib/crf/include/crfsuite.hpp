/*
 *      CRFsuite C++/SWIG API wrapper.
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

#ifndef __CRFSUITE_HPP__
#define __CRFSUITE_HPP__

#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>

#include <crfsuite.h>
#include "crfsuite_api.hpp"

namespace crfsuite
{

trainer::trainer()
{
    data = new crfsuite_data_t;
    if (data != NULL) {
        crfsuite_data_init(data);
        tr = NULL;
    }
}

trainer::~trainer()
{
    if (data != NULL) {
        if (data->labels != NULL) {
            data->labels->release(data->labels);
            data->labels = NULL;
        }

        if (data->attrs != NULL) {
            data->attrs->release(data->attrs);
            data->attrs = NULL;
        }

        crfsuite_data_finish(data);
        delete data;
        data = NULL;
    }

    if (tr != NULL) {
        tr->release(tr);
        tr = NULL;
    }
}

bool trainer::init(const std::string& type, const std::string& algorithm)
{
    int ret;

    // Build the trainer string ID.
    std::string tid = "train/";
    tid += type;
    tid += '/';
    tid += algorithm;

    // Create an instance of a trainer.
    ret = crfsuite_create_instance(tid.c_str(), (void**)&tr);
    if (!ret) {
        return false;
    }

    // Create an instance of attribute dictionary.
    ret = crfsuite_create_instance("dictionary", (void**)&data->attrs);
    if (!ret) {
        throw std::invalid_argument(" Failed to create a dictionary instance.");
    }

    // Create an instance of label dictionary.
    ret = crfsuite_create_instance("dictionary", (void**)&data->labels);
    if (!ret) {
        throw std::invalid_argument(" Failed to create a dictionary instance.");
    }

    // Set the callback function for receiving messages.
    tr->set_message_callback(tr, this, __logging_callback);

    return true;
}

void trainer::append_instance(const instance& inst)
{
    crfsuite_instance_t _inst;

    // Make sure |y| == |x|.
    if (inst.xseq.size() != inst.yseq.size()) {
        throw std::invalid_argument("The numbers of items and labels differ");
    }

    // Convert instance_type to crfsuite_instance_t.
    crfsuite_instance_init_n(&_inst, inst.xseq.size());
    for (size_t t = 0;t < inst.xseq.size();++t) {
        const item& item = inst.xseq[t];
        crfsuite_item_t* _item = &_inst.items[t];

        // Set the attributes in the item.
        crfsuite_item_init_n(_item, item.size());
        for (size_t i = 0;i < item.size();++i) {
            _item->contents[i].aid = data->attrs->get(data->attrs, item[i].attr.c_str());
            _item->contents[i].scale = (floatval_t)item[i].scale;
        }

        // Set the label of the item.
        _inst.labels[t] = data->labels->get(data->labels, inst.yseq[t].c_str());
    }
    _inst.group = inst.group;

    // Append the instance to the training set.
    crfsuite_data_append(data, &_inst);

    // Finish the instance.
    crfsuite_instance_finish(&_inst);
}

int trainer::train(const std::string& model, int holdout)
{
    int ret = tr->train(tr, data, model.c_str(), holdout);
    return ret;
}

void trainer::set_parameter(const std::string& name, const std::string& value)
{
    crfsuite_params_t* params = tr->params(tr);
    params->set(params, name.c_str(), value.c_str());
    params->release(params);
}

void trainer::receive_message(const std::string& msg)
{
}

int trainer::__logging_callback(void *instance, const char *format, va_list args)
{
    char buffer[65536];
    vsnprintf(buffer, sizeof(buffer)-1, format, args);
    reinterpret_cast<trainer*>(instance)->receive_message(buffer);
    return 0;
}



tagger::tagger()
{
    model = NULL;
}

tagger::~tagger()
{
    this->close();
}

bool tagger::open(const std::string& name)
{
    int ret;

    this->close();

    // Open the model file.
    if ((ret = crfsuite_create_instance_from_file(name.c_str(), (void**)&model))) {
        return false;
    }

    return true;
}

void tagger::close()
{
    if (model != NULL) {
        model->release(model);
        model = NULL;
    }
}

labels tagger::tag(const instance& inst)
{
    int ret;
    labels yseq;
    crfsuite_instance_t _inst;
    crfsuite_tagger_t *tag = NULL;
    crfsuite_dictionary_t *attrs = NULL, *labels = NULL;

    // Obtain the dictionary interface representing the labels in the model.
    if ((ret = model->get_labels(model, &labels))) {
        throw std::invalid_argument("Failed to obtain the dictionary interface for labels");
    }

    // Obtain the dictionary interface representing the attributes in the model.
    if ((ret = model->get_attrs(model, &attrs))) {
        throw std::invalid_argument("Failed to obtain the dictionary interface for attributes");
    }

    // Obtain the tagger interface.
    if ((ret = model->get_tagger(model, &tag))) {
        throw std::invalid_argument("Failed to obtain the tagger interface");
    }

    // Build an instance.
    crfsuite_instance_init_n(&_inst, inst.xseq.size());
    for (size_t t = 0;t < inst.xseq.size();++t) {
        const item& item = inst.xseq[t];
        crfsuite_item_t* _item = &_inst.items[t];

        // Set the attributes in the item.
        crfsuite_item_init(_item);
        for (size_t i = 0;i < item.size();++i) {
            int aid = attrs->to_id(attrs, item[i].attr.c_str());
            if (0 <= aid) {
                crfsuite_content_t cont;
                crfsuite_content_set(&cont, aid, item[i].scale);
                crfsuite_item_append_content(_item, &cont);
            }
        }
    }

    floatval_t score;
    if ((ret = tag->tag(tag, &_inst, _inst.labels, &score))) {
        crfsuite_instance_finish(&_inst);
        throw std::invalid_argument("Failed to tag the instance.");
    }

    yseq.resize(inst.xseq.size());
    for (size_t t = 0;t < inst.xseq.size();++t) {
        const char *label = NULL;
        labels->to_string(labels, _inst.labels[t], &label);
        yseq[t] = label;
        labels->free(labels, label);
    }

    // Finish the instance.
    crfsuite_instance_finish(&_inst);

    return yseq;
}

};

#endif/*__CRFSUITE_HPP__*/
