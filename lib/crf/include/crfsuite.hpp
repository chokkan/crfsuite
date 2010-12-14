#include <vector>
#include <string>
#include <stdexcept>

#include <crfsuite.h>

namespace crfsuite
{

/**
 * Tuple of attribute and its value.
 */
struct attribute_type
{
    /// Attribute.
    std::string attr;
    /// Attribute value (weight).
    floatval_t  scale;
};

typedef std::vector<attribute_type> item_type;
typedef std::vector<item_type>  item_sequence_type;
typedef std::vector<std::string> label_sequence_type;

/**
 * Instance (sequence of items and their labels).
 */
struct instance_type
{
    /// Item sequence.
    item_sequence_type  xseq;
    /// Label sequence.
    label_sequence_type yseq;
    /// Group number.
    int group;
};

/**
 * Exceptions from trainer.
 */
class trainer_error : public std::invalid_argument {
public:
    /**
     * Constructs an instance.
     *  @param  msg         The error message.
     */
    explicit trainer_error(const char *msg) : invalid_argument(msg)
    {
    }
};

/**
 * Trainer class.
 */
class trainer {
protected:
    crf_data_t data;
    crf_trainer_t *tr;

public:
    /**
     * Constructs an instance.
     */
    trainer()
    {
        crf_data_init(&data);
        tr = NULL;
    }

    /**
     * Destructs an instance.
     */
    virtual ~trainer()
    {
        crf_data_finish(&data);

        if (data.labels != NULL) {
            data.labels->release(data.labels);
            data.labels = NULL;
        }

        if (data.attrs != NULL) {
            data.attrs->release(data.attrs);
            data.attrs = NULL;
        }

        if (tr != NULL) {
            tr->release(tr);
            tr = NULL;
        }
    }

    /**
     * Initialize the trainer with specified CRF type and training algorithm.
     *  @param  type            The name of the CRF type.
     *  @param  algorithm       The name of the training algorithm.
     *  @return bool            \c true if the CRF type and training algorithm
     *                          are available, \c false otherwise.
     */
    bool init(const std::string& type, const std::string& algorithm)
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
        ret = crf_create_instance("dictionary", (void**)&data.attrs);
        if (!ret) {
            throw trainer_error(" Failed to create a dictionary instance.");
        }

        // Create an instance of label dictionary.
        ret = crf_create_instance("dictionary", (void**)&data.labels);
        if (!ret) {
            throw trainer_error(" Failed to create a dictionary instance.");
        }

        // Set the callback function for receiving messages.
        tr->set_message_callback(tr, this, __logging_callback);

        return true;
    }

    /**
     * Appends an instance to the data set.
     *  @param  inst        The instance to be appended.
     */
    void append_instance(const instance_type& inst)
    {
        crf_instance_t _inst;

        // Make sure |y| == |x|.
        if (inst.xseq.size() != inst.yseq.size()) {
            throw trainer_error("The numbers of items and labels differ");
        }

        // Convert instance_type to crf_instance_t.
        crf_instance_init_n(&_inst, inst.xseq.size());
        for (size_t t = 0;t < inst.xseq.size();++t) {
            const item_type& item = inst.xseq[t];
            crf_item_t* _item = &_inst.items[t];

            // Set the attributes in the item.
            crf_item_init_n(_item, item.size());
            for (size_t i = 0;i < item.size();++i) {
                _item->contents[i].aid = data.attrs->get(data.attrs, item[i].attr.c_str());
                _item->contents[i].scale = item[i].scale;
            }

            // Set the label of the item.
            _inst.labels[t] = data.labels->get(data.labels, inst.yseq[t].c_str());
        }
        _inst.group = inst.group;

        // Append the instance to the training set.
        crf_data_append(&data, &_inst);

        // Finish the instance.
        crf_instance_finish(&_inst);
    }

    /**
     * Runs the training algorithm.
     *  @param  model       The filename to which the obtained model is stored.
     *  @param  holdout     The group number of holdout evaluation.
     *  @return int         The status code.
     */
    int train(const std::string& model, int holdout)
    {
        int ret = tr->train(tr, &data, model.c_str(), holdout);
        return ret;
    }

    /**
     * Sets the training parameter.
     *  @param  name        The parameter name.
     *  @param  value       The value of the parameter.
     */
    void set_parameter(const std::string& name, const std::string& value)
    {
        crf_params_t* params = tr->params(tr);
        params->set(params, name.c_str(), value.c_str());
        params->release(params);
    }

    /**
     * Receives messages from the training algorithm.
     *  Override this member function in the inheritance class if
     *  @param  msg         The message
     */
    virtual void receive_message(const std::string& msg)
    {
    }

protected:
    static int __logging_callback(void *instance, const char *format, va_list args)
    {
        char buffer[65536];
        _vsnprintf(buffer, sizeof(buffer)-1, format, args);
        reinterpret_cast<trainer*>(instance)->receive_message(buffer);
        return 0;
    }
};

/**
 * Exceptions from tagger.
 */
class tagger_error : public std::invalid_argument {
public:
    /**
     * Constructs an instance.
     *  @param  msg         The error message.
     */
    tagger_error(const char *msg) : invalid_argument(msg)
    {
    }
};

/**
 * Tagger class.
 */
class tagger
{
protected:
    crf_model_t *model;

public:
    /**
     * Constructs a tagger.
     */
    tagger()
    {
        model = NULL;
    }

    /**
     * Destructs a tagger.
     */
    virtual ~tagger()
    {
        if (model != NULL) {
            model->release(model);
            model = NULL;
        }
    }

    /**
     * Tags an instance.
     *  @param  inst        The instance to be tagged.
     *  @param  yseq        The label sequence to which the tagging result is
     *                      stored.
     */
    void tag(const instance_type& inst, label_sequence_type& yseq)
    {
        int ret;
        crf_instance_t _inst;
        crf_tagger_t *tagger = NULL;
        crf_dictionary_t *attrs = NULL, *labels = NULL;

        // Obtain the dictionary interface representing the labels in the model.
        if (ret = model->get_labels(model, &labels)) {
            throw tagger_error("Failed to obtain the dictionary interface for labels");
        }

        // Obtain the dictionary interface representing the attributes in the model.
        if (ret = model->get_attrs(model, &attrs)) {
            throw tagger_error("Failed to obtain the dictionary interface for attributes");
        }

        // Obtain the tagger interface.
        if (ret = model->get_tagger(model, &tagger)) {
            throw tagger_error("Failed to obtain the tagger interface");
        }

        // Build an instance.
        crf_instance_init_n(&_inst, inst.xseq.size());
        for (size_t t = 0;t < inst.xseq.size();++t) {
            const item_type& item = inst.xseq[t];
            crf_item_t* _item = &_inst.items[t];

            // Set the attributes in the item.
            crf_item_init_n(_item, item.size());
            for (size_t i = 0;i < item.size();++i) {
                _item->contents[i].aid = attrs->to_id(attrs, item[i].attr.c_str());
                _item->contents[i].scale = item[i].scale;
            }
        }

        floatval_t score;
        if (ret = tagger->tag(tagger, &_inst, _inst.labels, &score)) {
            crf_instance_finish(&_inst);
            throw tagger_error("Failed to tag the instance.");
        }

        yseq.resize(inst.xseq.size());
        for (size_t t = 0;t < inst.xseq.size();++t) {
            const char *label = NULL;
            labels->to_string(labels, _inst.labels[t], &label);
            yseq[t] = label;
            labels->free(labels, label);
        }

        // Finish the instance.
        crf_instance_finish(&_inst);
    }
};

};
