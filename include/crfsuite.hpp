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
#include <sstream>

#include <crfsuite.h>
#include "crfsuite_api.hpp"

namespace CRFSuite
{

Trainer::Trainer()
{
    data = new crfsuite_data_t;
    if (data != NULL) {
        crfsuite_data_init(data);
    }
    tr = NULL;
}

Trainer::~Trainer()
{
    if (data != NULL) {
        clear();
        delete data;
        data = NULL;
    }
    if (tr != NULL) {
        tr->release(tr);
        tr = NULL;
    }
}

void Trainer::init()
{
    // Create an instance of attribute dictionary.
    if (data->attrs == NULL) {
        int ret = crfsuite_create_instance("dictionary", (void**)&data->attrs);
        if (!ret) {
            throw std::invalid_argument("Failed to create a dictionary instance for attributes.");
        }
    }

    // Create an instance of label dictionary.
    if (data->labels == NULL) {
        int ret = crfsuite_create_instance("dictionary", (void**)&data->labels);
        if (!ret) {
            throw std::invalid_argument("Failed to create a dictionary instance for labels.");
        }
    }
}

void Trainer::clear()
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
        crfsuite_data_init(data);
    }
}

void Trainer::append(const ItemSequence& xseq, const LabelSequence& yseq, int group)
{
    // Create dictionary objects if necessary.
    if (data->attrs == NULL || data->labels == NULL) {
        init();
    }

    // Make sure |y| == |x|.
    if (xseq.size() != yseq.size()) {
        std::stringstream ss;
        ss << "The numbers of items and labels differ: |x| = " << xseq.size() << ", |y| = " << yseq.size();
        throw std::invalid_argument(ss.str());
    }

    // Convert instance_type to crfsuite_instance_t.
    crfsuite_instance_t _inst;
    crfsuite_instance_init_n(&_inst, xseq.size());
    for (size_t t = 0;t < xseq.size();++t) {
        const Item& item = xseq[t];
        crfsuite_item_t* _item = &_inst.items[t];

        // Set the attributes in the item.
        crfsuite_item_init_n(_item, item.size());
        for (size_t i = 0;i < item.size();++i) {
            _item->contents[i].aid = data->attrs->get(data->attrs, item[i].attr.c_str());
            _item->contents[i].scale = (floatval_t)item[i].value;
        }

        // Set the label of the item.
        _inst.labels[t] = data->labels->get(data->labels, yseq[t].c_str());
    }
    _inst.group = group;

    // Append the instance to the training set.
    crfsuite_data_append(data, &_inst);

    // Finish the instance.
    crfsuite_instance_finish(&_inst);
}

bool Trainer::select(const std::string& algorithm, const std::string& type)
{
    int ret;

    // Release the trainer if it is already initialized.
    if (tr != NULL) {
        tr->release(tr);
        tr = NULL;
    }

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

    // Set the callback function for receiving messages.
    tr->set_message_callback(tr, this, __logging_callback);

    return true;
}

int Trainer::train(const std::string& model, int holdout)
{
    int ret;

    // Run the training algorithm.
    ret = tr->train(tr, data, model.c_str(), holdout);

    return ret;
}

StringList Trainer::params()
{
    StringList pars;
    crfsuite_params_t* params = tr->params(tr);
    int n = params->num(params);
    for (int i = 0;i < n;++i) {
        char *name = NULL;
        params->name(params, i, &name);
        pars.push_back(name);
        params->free(params, name);
    }
    return pars;
}

void Trainer::set(const std::string& name, const std::string& value)
{
    crfsuite_params_t* params = tr->params(tr);
    if (params->set(params, name.c_str(), value.c_str()) != 0) {
        std::stringstream ss;
        ss << "Parameter not found: " << name << " = " << value;
        params->release(params);
        throw std::invalid_argument(ss.str());
    }
    params->release(params);
}

std::string Trainer::get(const std::string& name)
{
    std::string value;
    char *_value = NULL;
    crfsuite_params_t* params = tr->params(tr);
    if (params->get(params, name.c_str(), &_value) != 0) {
        std::stringstream ss;
        ss << "Parameter not found: " << name << " = " << value;
        params->release(params);
        throw std::invalid_argument(ss.str());
    }
    value = _value;
    params->free(params, _value);
    params->release(params);
    return value;
}

std::string Trainer::help(const std::string& name)
{
    std::string str;
    crfsuite_params_t* params = tr->params(tr);
    char *_str = NULL;
    params->help(params, name.c_str(), NULL, &_str);
    str = _str;
    params->free(params, _str);
    params->release(params);
    return str;
}

void Trainer::message(const std::string& msg)
{
}

int Trainer::__logging_callback(void *instance, const char *format, va_list args)
{
    char buffer[65536];
    vsnprintf(buffer, sizeof(buffer)-1, format, args);
    reinterpret_cast<Trainer*>(instance)->message(buffer);
    return 0;
}



Tagger::Tagger()
{
    model = NULL;
}

Tagger::~Tagger()
{
    this->close();
}

bool Tagger::open(const std::string& name)
{
    int ret;

    // Close the model if it is already opened.
    this->close();

    // Open the model file.
    if ((ret = crfsuite_create_instance_from_file(name.c_str(), (void**)&model))) {
        return false;
    }

    return true;
}

void Tagger::close()
{
    if (model != NULL) {
        model->release(model);
        model = NULL;
    }
}

LabelSequence Tagger::tag(const ItemSequence& xseq)
{
    int ret;
    LabelSequence yseq;
    crfsuite_instance_t _inst;
    crfsuite_tagger_t *tag = NULL;
    crfsuite_dictionary_t *attrs = NULL, *labels = NULL;

    // Obtain the dictionary interface representing the labels in the model.
    if ((ret = model->get_labels(model, &labels))) {
        throw std::invalid_argument("Failed to obtain the dictionary interface for labels");
    }

    // Obtain the dictionary interface representing the attributes in the model.
    if ((ret = model->get_attrs(model, &attrs))) {
        labels->release(labels);
        throw std::invalid_argument("Failed to obtain the dictionary interface for attributes");
    }

    // Obtain the tagger interface.
    if ((ret = model->get_tagger(model, &tag))) {
        attrs->release(attrs);
        labels->release(labels);
        throw std::invalid_argument("Failed to obtain the tagger interface");
    }

    // Build an instance.
    crfsuite_instance_init_n(&_inst, xseq.size());
    for (size_t t = 0;t < xseq.size();++t) {
        const Item& item = xseq[t];
        crfsuite_item_t* _item = &_inst.items[t];

        // Set the attributes in the item.
        crfsuite_item_init(_item);
        for (size_t i = 0;i < item.size();++i) {
            int aid = attrs->to_id(attrs, item[i].attr.c_str());
            if (0 <= aid) {
                crfsuite_content_t cont;
                crfsuite_content_set(&cont, aid, item[i].value);
                crfsuite_item_append_content(_item, &cont);
            }
        }
    }

    floatval_t score;
    if ((ret = tag->tag(tag, &_inst, _inst.labels, &score))) {
        crfsuite_instance_finish(&_inst);
        tag->release(tag);
        attrs->release(attrs);
        labels->release(labels);
        throw std::invalid_argument("Failed to tag the instance.");
    }

    yseq.resize(xseq.size());
    for (size_t t = 0;t < xseq.size();++t) {
        const char *label = NULL;
        if (labels->to_string(labels, _inst.labels[t], &label) != 0) {
            crfsuite_instance_finish(&_inst);
            tag->release(tag);
            attrs->release(attrs);
            labels->release(labels);
            throw std::runtime_error("Failed to convert a label ID to string.");
        }
        yseq[t] = label;
        labels->free(labels, label);
    }

    crfsuite_instance_finish(&_inst);
    tag->release(tag);
    attrs->release(attrs);
    labels->release(labels);

    return yseq;
}

LabelSequence Tagger::labels()
{
    int ret;
    LabelSequence lseq;
    crfsuite_dictionary_t *labels = NULL;

    // Obtain the dictionary interface representing the labels in the model.
    if ((ret = model->get_labels(model, &labels))) {
        throw std::invalid_argument("Failed to obtain the dictionary interface for labels");
    }

    // Collect all label strings to lseq.
    for (int i = 0;i < labels->num(labels);++i) {
        const char *label = NULL;
        if (labels->to_string(labels, i, &label) != 0) {
            labels->release(labels);
            throw std::runtime_error("Failed to convert a label ID to string.");
        }
        lseq.push_back(label);
        labels->free(labels, label);
    }

    labels->release(labels);

    return lseq;
}

};

#endif/*__CRFSUITE_HPP__*/

