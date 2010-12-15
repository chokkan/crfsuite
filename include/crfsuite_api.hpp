/*
 *      CRFsuite C++/SWIG API.
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

#ifndef __CRFSUITE_API_HPP__
#define __CRFSUITE_API_HPP__

#include <vector>
#include <string>
#include <stdexcept>

#ifndef __CRFSUITE_H__

#ifdef  __cplusplus
extern "C" {
#endif/*__cplusplus*/

struct tag_crfsuite_model;
typedef struct tag_crfsuite_model crfsuite_model_t;

struct tag_crfsuite_data;
typedef struct tag_crfsuite_data crfsuite_data_t;

struct tag_crfsuite_trainer;
typedef struct tag_crfsuite_trainer crfsuite_trainer_t;

struct tag_crfsuite_tagger;
typedef struct tag_crfsuite_tagger crfsuite_tagger_t;

struct tag_crfsuite_dictionary;
typedef struct tag_crfsuite_dictionary crfsuite_dictionary_t;

struct tag_crfsuite_params;
typedef struct tag_crfsuite_params crfsuite_params_t;

#ifdef  __cplusplus
}
#endif/*__cplusplus*/

#endif/*__CRFSUITE_H__*/

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
    crfsuite_data_t *data;
    crfsuite_trainer_t *tr;

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
    crfsuite_model_t *model;

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

#endif/*__CRFSUITE_API_HPP__*/
