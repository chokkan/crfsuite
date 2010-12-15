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
    data = new crf_data_t;
    if (data != NULL) {
        crf_data_init(data);
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

        crf_data_finish(data);
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
    ret = crf_create_instance(tid.c_str(), (void**)&tr);
    if (!ret) {
        return false;
    }

    // Create an instance of attribute dictionary.
    ret = crf_create_instance("dictionary", (void**)&data->attrs);
    if (!ret) {
        throw std::invalid_argument(" Failed to create a dictionary instance.");
    }

    // Create an instance of label dictionary.
    ret = crf_create_instance("dictionary", (void**)&data->labels);
    if (!ret) {
        throw std::invalid_argument(" Failed to create a dictionary instance.");
    }

    // Set the callback function for receiving messages.
    tr->set_message_callback(tr, this, __logging_callback);

    return true;
}

void trainer::append_instance(const instance& inst)
{
    crf_instance_t _inst;

    // Make sure |y| == |x|.
    if (inst.xseq.size() != inst.yseq.size()) {
        throw std::invalid_argument("The numbers of items and labels differ");
    }

    // Convert instance_type to crf_instance_t.
    crf_instance_init_n(&_inst, inst.xseq.size());
    for (size_t t = 0;t < inst.xseq.size();++t) {
        const item& item = inst.xseq[t];
        crf_item_t* _item = &_inst.items[t];

        // Set the attributes in the item.
        crf_item_init_n(_item, item.size());
        for (size_t i = 0;i < item.size();++i) {
            _item->contents[i].aid = data->attrs->get(data->attrs, item[i].attr.c_str());
            _item->contents[i].scale = (floatval_t)item[i].scale;
        }

        // Set the label of the item.
        _inst.labels[t] = data->labels->get(data->labels, inst.yseq[t].c_str());
    }
    _inst.group = inst.group;

    // Append the instance to the training set.
    crf_data_append(data, &_inst);

    // Finish the instance.
    crf_instance_finish(&_inst);
}

int trainer::train(const std::string& model, int holdout)
{
    int ret = tr->train(tr, data, model.c_str(), holdout);
    return ret;
}

void trainer::set_parameter(const std::string& name, const std::string& value)
{
    crf_params_t* params = tr->params(tr);
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
    if ((ret = crf_create_instance_from_file(name.c_str(), (void**)&model))) {
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
    crf_instance_t _inst;
    crf_tagger_t *tag = NULL;
    crf_dictionary_t *attrs = NULL, *labels = NULL;

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

    std::cout << "OK1" << std::endl;
    std::cout << "len = " << inst.xseq.size() << std::endl;

    // Build an instance.
    crf_instance_init_n(&_inst, inst.xseq.size());
    for (size_t t = 0;t < inst.xseq.size();++t) {
        const item& item = inst.xseq[t];
        crf_item_t* _item = &_inst.items[t];

        // Set the attributes in the item.
        crf_item_init(_item);
        for (size_t i = 0;i < item.size();++i) {
            int aid = attrs->to_id(attrs, item[i].attr.c_str());
            if (0 <= aid) {
                crf_content_t cont;
                crf_content_set(&cont, aid, item[i].scale);
                crf_item_append_content(_item, &cont);
            }
        }
    }

    std::cout << "OK2" << std::endl;

    printf("0x%p\n", tag);
    printf("0x%p\n", tag->tag);
    printf("0x%p\n", attrs);
    printf("0x%p\n", attrs->to_id);
    printf("0x%p\n", &_inst);
    printf("0x%p\n", _inst.labels);

    floatval_t score;
    if ((ret = tag->tag(tag, &_inst, _inst.labels, &score))) {
        std::cout << "OK3" << std::endl;
        crf_instance_finish(&_inst);
        throw std::invalid_argument("Failed to tag the instance.");
    }

    std::cout << "OK4" << std::endl;

    yseq.resize(inst.xseq.size());
    for (size_t t = 0;t < inst.xseq.size();++t) {
        const char *label = NULL;
        labels->to_string(labels, _inst.labels[t], &label);
        yseq[t] = label;
        labels->free(labels, label);
    }

    // Finish the instance.
    crf_instance_finish(&_inst);

    return yseq;
}

};

