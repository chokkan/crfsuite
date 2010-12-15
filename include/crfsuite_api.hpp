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

namespace CRFSuite
{

/**
 * Tuple of attribute and its value.
 */
class Attribute
{
public:
    /// Attribute.
    std::string attr;
    /// Attribute value (weight).
    double value;

    /**
     * Constructs an attribute with the default name and value.
     */
    Attribute() : value(1.)
    {
    }

    /**
     * Constructs an attribute with the default value.
     *  @param  name        The attribute name.
     */
    Attribute(const std::string& name) : attr(name), value(1.)
    {
    }

    /**
     * Constructs an attribute.
     *  @param  name        The attribute name.
     *  @aram   v           The attribute value.
     */
    Attribute(const std::string& name, double v) : attr(name), value(v) {}
};

/// Type of an item (attribute vector).
typedef std::vector<Attribute> Item;

/// Type of an item sequence.
typedef std::vector<Item>  ItemSequence;

/// Type of a label sequence.
typedef std::vector<std::string> LabelSequence;



/**
 * Trainer class.
 */
class Trainer {
protected:
    crfsuite_data_t *data;
    crfsuite_trainer_t *tr;

public:
    /**
     * Constructs an instance.
     */
    Trainer();

    /**
     * Destructs an instance.
     */
    virtual ~Trainer();

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
     *  @param  xseq            The item sequence of the instance.
     *  @param  yseq            The label sequence of the instance.
     *  @param  group           The group number of the instance.
     */
    void append(const ItemSequence& xseq, const LabelSequence& yseq, int group);

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
    void set(const std::string& name, const std::string& value);

    /**
     * Receives messages from the training algorithm.
     *  Override this member function in the inheritance class if
     *  @param  msg         The message
     */
    virtual void message(const std::string& msg);

protected:
    static int __logging_callback(void *userdata, const char *format, va_list args);
};

/**
 * Tagger class.
 */
class Tagger
{
protected:
    crfsuite_model_t *model;

public:
    /**
     * Constructs a tagger.
     */
    Tagger();

    /**
     * Destructs a tagger.
     */
    virtual ~Tagger();

    /**
     * Opens a model file.
     *  @param  model       The file name of the model file.
     *  @return bool        \c true if the model file is successfully opened,
     *                      \c false otherwise.
     */
    bool open(const std::string& name);

    /**
     * Closes the model.
     */
    void close();

    /**
     * Tags an instance.
     *  @param  xseq            The item sequence to be tagged..
     *  @return LabelSequence   The label sequence predicted.
     */
    LabelSequence tag(const ItemSequence& xseq);

    /**
     * Obtains the list of labels.
     *  @return LabelSequence   The list of labels in the model.
     */
    LabelSequence labels();
};

};

#endif/*__CRFSUITE_API_HPP__*/
