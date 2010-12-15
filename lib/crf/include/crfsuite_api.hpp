#ifndef __EXPORT_H__
#define __EXPORT_H__

#include <vector>
#include <string>
#include <stdexcept>

struct tag_crf_model;
typedef struct tag_crf_model crf_model_t;

struct tag_crf_data;
typedef struct tag_crf_data crf_data_t;

struct tag_crf_trainer;
typedef struct tag_crf_trainer crf_trainer_t;

struct tag_crf_tagger;
typedef struct tag_crf_tagger crf_tagger_t;

struct tag_crf_dictionary;
typedef struct tag_crf_dictionary crf_dictionary_t;

struct tag_crf_params;
typedef struct tag_crf_params crf_params_t;

namespace crfsuite
{

/**
 * Tuple of attribute and its value.
 */
class feature
{
public:
    /// Attribute.
    std::string attr;
    /// Attribute value (weight).
    double scale;

    /// Default constructor.
    feature() : scale(1.) {}

    feature(const std::string& _attr, double _scale) : attr(_attr), scale(_scale) {}
};

typedef std::vector<feature> item;
typedef std::vector<item>  items;
typedef std::vector<std::string> labels;

/**
 * Instance (sequence of items and their labels).
 */
struct instance
{
    /// Item sequence.
    items  xseq;
    /// Label sequence.
    labels yseq;
    /// Group number.
    int group;
};



/**
 * Trainer class.
 */
class trainer {
protected:
    crf_data_t *data;
    crf_trainer_t *tr;

public:
    /**
     * Constructs an instance.
     */
    trainer();

    /**
     * Destructs an instance.
     */
    virtual ~trainer();

    /**
     * Initialize the trainer with specified type and training algorithm.
     *  @param  type            The name of the CRF type.
     *  @param  algorithm       The name of the training algorithm.
     *  @return bool            \c true if the CRF type and training
     *                          algorithm are available,
     *                          \c false otherwise.
     */
    bool init(const std::string& type, const std::string& algorithm);

    /**
     * Appends an instance to the data set.
     *  @param  inst            The instance to be appended.
     */
    void append_instance(const instance& inst);

    /**
     * Runs the training algorithm.
     *  @param  model       The filename to which the obtained model is stored.
     *  @param  holdout     The group number of holdout evaluation.
     *  @return int         The status code.
     */
    int train(const std::string& model, int holdout);

    /**
     * Sets the training parameter.
     *  @param  name        The parameter name.
     *  @param  value       The value of the parameter.
     */
    void set_parameter(const std::string& name, const std::string& value);

    /**
     * Receives messages from the training algorithm.
     *  Override this member function in the inheritance class if
     *  @param  msg         The message
     */
    virtual void receive_message(const std::string& msg);

protected:
    static int __logging_callback(void *instance, const char *format, va_list args);
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
    tagger();

    /**
     * Destructs a tagger.
     */
    virtual ~tagger();

    /**
     * Opens a model file.
     *  @param  model       The file name of the model file.
     *  @return bool        \c true if the model file is successfully opened,
     *                      \c false otherwise.
     */
    bool open(const std::string& name);

    /**
     * Closes a model file.
     */
    void close();

    /**
     * Tags an instance.
     *  @param  inst        The instance to be tagged.
     *  @param  yseq        The label sequence to which the tagging result is
     *                      stored.
     */
    labels tag(const instance& inst);
};

};

#endif/*__EXPORT_H__*/

