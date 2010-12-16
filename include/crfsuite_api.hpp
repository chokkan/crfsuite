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

#include <string>
#include <stdexcept>
#include <vector>

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

/// Type of a string list.
typedef std::vector<std::string> StringList;



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
     * Removes all instances in the data set.
     */
    void clear();

    /**
     * Appends an instance to the data set.
     *  @param  xseq        The item sequence of the instance.
     *  @param  yseq        The label sequence of the instance.
     *  @param  group       The group number of the instance.
     */
    void append(const ItemSequence& xseq, const StringList& yseq, int group);

    /**
     * Initializes the training algorithm.
     *  @param  algorithm   The name of the training algorithm.
     *  @param  type        The name of the CRF type.
     *  @return bool        \c true if the training algorithm is successfully
     *                      initialized, \c false otherwise.
     */
    bool select(const std::string& algorithm, const std::string& type);

    /**
     * Runs the training algorithm.
     *  @param  model       The filename to which the obtained model is stored.
     *  @param  holdout     The group number of holdout evaluation.
     *  @return int         The status code.
     */
    int train(const std::string& model, int holdout);

    /**
     * Obtains the list of parameters.
     *  @return StringList  The list of parameters available for the current
     *                      training algorithm.
     */
    StringList params();

    /**
     * Sets the training parameter.
     *  @param  name        The parameter name.
     *  @param  value       The value of the parameter.
     */
    void set(const std::string& name, const std::string& value);

    /**
     * Gets the value of a training parameter.
     *  @param  name        The parameter name.
     *  @return std::string The value of the parameter.
     */
    std::string get(const std::string& name);

    /**
     * Gets the description of a training parameter.
     *  @param  name        The parameter name.
     *  @return std::string The description (help message) of the parameter.
     */
    std::string help(const std::string& name);

    /**
     * Receives messages from the training algorithm.
     *  Override this member function in the inheritance class if
     *  @param  msg         The message
     */
    virtual void message(const std::string& msg);

protected:
    void init();
    static int __logging_callback(void *userdata, const char *format, va_list args);
};

/**
 * Tagger class.
 */
class Tagger
{
protected:
    crfsuite_model_t *model;
    crfsuite_tagger_t *tagger;

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
     * Obtains the list of labels.
     *  @return StringList  The list of labels in the model.
     */
    StringList labels();

    /**
     * Predicts the label sequence for the item sequence.
     *  @param  xseq        The item sequence to be tagged.
     *  @return StringList  The label sequence predicted.
     */
    StringList tag(const ItemSequence& xseq);

    /**
     * Sets an item sequence.
     *  @param  xseq        The item sequence to be tagged    
     */
    void set(const ItemSequence& xseq);

    /**
     * Finds the Viterbi label sequence for the item sequence.
     *  @return StringList  The label sequence predicted.
     */
    StringList viterbi();

    /**
     * Computes the probability of the label sequence.
     *  @param  yseq        The label sequence.
     */
    double probability(const StringList& yseq);

    /**
     * Computes the marginal probability of the label.
     *  @param  y           The label.
     *  @param  t           The position of the label.
     */
    double marginal(const std::string& y, const int t);
};

std::string version();

};

#endif/*__CRFSUITE_API_HPP__*/
