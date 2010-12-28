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

/** 
 * \addtogroup crfsuite_hpp_api CRFSuite C++/SWIG API
 * @{
 *
 *  The CRFSuite C++/SWIG API.
 */

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
     *  @aram   val         The attribute value.
     */
    Attribute(const std::string& name, double val) : attr(name), value(val)
    {
    }
};

/// Type of an item (equivalent to an attribute vector) in a sequence.
typedef std::vector<Attribute> Item;

/// Type of an item sequence (equivalent to item vector).
typedef std::vector<Item>  ItemSequence;

/// Type of a string list.
typedef std::vector<std::string> StringList;



/**
 * The trainer class.
 *  This class maintains a data set for training, and provides an interface
 *  to various graphical models and training algorithms. The standard
 *  procedure for implementing a trainer is:
 *  - create a class by inheriting this class
 *  - overwrite message() function to receive messages of training progress
 *  - call append() to append item/label sequences to the training set
 *  - call select() to specify a graphical model and an algorithm
 *  - call set() to configure parameters specific to the model and algorithm
 *  - call train() to start a training process with the current setting
 */
class Trainer {
protected:
    crfsuite_data_t *data;
    crfsuite_trainer_t *tr;
    
public:
    /**
     * Constructs a trainer.
     */
    Trainer();

    /**
     * Destructs a trainer.
     */
    virtual ~Trainer();

    /**
     * Removes all instances in the data set.
     */
    void clear();

    /**
     * Appends an instance (item/label sequence) to the data set.
     *  @param  xseq        The item sequence of the instance.
     *  @param  yseq        The label sequence of the instance. The number
     *                      of elements in yseq must be identical to that
     *                      in xseq.
     *  @param  group       The group number of the instance.
     *  @throw  std::invalid_argument   Arguments xseq and yseq are invalid.
     *  @throw  std::runtime_error      Out of memory.
     */
    void append(const ItemSequence& xseq, const StringList& yseq, int group);

    /**
     * Initializes the training algorithm.
     *  @param  algorithm   The name of the training algorithm.
     *  @param  type        The name of the graphical model.
     *  @return bool        \c true if the training algorithm is successfully
     *                      initialized, \c false otherwise.
     */
    bool select(const std::string& algorithm, const std::string& type);

    /**
     * Runs the training algorithm.
     *  This function starts the training algorithm with the data set given
     *  by append() function. After starting the training process, the 
     *  training algorithm invokes the virtual function message() to report
     *  the progress of the training process.
     *  @param  model       The filename to which the trained model is stored.
     *                      If this value is empty, this function does not
     *                      write out a model file.
     *  @param  holdout     The group number of holdout evaluation. The
     *                      instances with this group number will not be used
     *                      for training, but for holdout evaluation. Specify
     *                      \c -1 to use all instances for training.
     *  @return int         The status code.
     */
    int train(const std::string& model, int holdout);

    /**
     * Obtains the list of parameters.
     *  This function returns the list of parameter names available for the
     *  graphical model and training algorithm specified by select() function.
     *  @return StringList  The list of parameters available for the current
     *                      graphical model and training algorithm.
     */
    StringList params();

    /**
     * Sets a training parameter.
     *  This function sets a parameter value for the graphical model and
     *  training algorithm specified by select() function.
     *  @param  name        The parameter name.
     *  @param  value       The value of the parameter.
     *  @throw  std::invalid_argument   The parameter is not found.
     */
    void set(const std::string& name, const std::string& value);

    /**
     * Gets the value of a training parameter.
     *  This function gets a parameter value for the graphical model and
     *  training algorithm specified by select() function.
     *  @param  name        The parameter name.
     *  @return std::string The value of the parameter.
     *  @throw  std::invalid_argument   The parameter is not found.
     */
    std::string get(const std::string& name);

    /**
     * Gets the description of a training parameter.
     *  This function obtains the help message for the parameter specified
     *  by the name. The graphical model and training algorithm must be
     *  selected by select() function before calling this function.
     *  @param  name        The parameter name.
     *  @return std::string The description (help message) of the parameter.
     */
    std::string help(const std::string& name);

    /**
     * Receives messages from the training algorithm.
     *  Override this member function to receive messages of the training
     *  process.
     *  @param  msg         The message
     */
    virtual void message(const std::string& msg);

protected:
    void init();
    static int __logging_callback(void *userdata, const char *format, va_list args);
};

/**
 * The tagger class.
 *  This class provides the functionality for predicting label sequences for
 *  input sequences using a model.
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
     *                      \c false otherwise (e.g., when the mode file is
     *                      not found).
     *  @throw  std::runtime_error      An internal error in the model.
     */
    bool open(const std::string& name);

    /**
     * Closes the model.
     */
    void close();

    /**
     * Obtains the list of labels.
     *  @return StringList  The list of labels in the model.
     *  @throw  std::invalid_argument   A model is not opened.
     *  @throw  std::runtime_error      An internal error.
     */
    StringList labels();

    /**
     * Predicts the label sequence for the item sequence.
     *  @param  xseq        The item sequence to be tagged.
     *  @return StringList  The label sequence predicted.
     *  @throw  std::invalid_argument   A model is not opened.
     *  @throw  std::runtime_error      An internal error.
     */
    StringList tag(const ItemSequence& xseq);

    /**
     * Sets an item sequence.
     *  @param  xseq        The item sequence to be tagged    
     *  @throw  std::invalid_argument   A model is not opened.
     *  @throw  std::runtime_error      An internal error.
     */
    void set(const ItemSequence& xseq);

    /**
     * Finds the Viterbi label sequence for the item sequence.
     *  @return StringList  The label sequence predicted.
     *  @throw  std::invalid_argument   A model is not opened.
     *  @throw  std::runtime_error      An internal error.
     */
    StringList viterbi();

    /**
     * Computes the probability of the label sequence.
     *  @param  yseq        The label sequence.
     *  @throw  std::invalid_argument   A model is not opened.
     *  @throw  std::runtime_error      An internal error.
     */
    double probability(const StringList& yseq);

    /**
     * Computes the marginal probability of the label.
     *  @param  y           The label.
     *  @param  t           The position of the label.
     *  @throw  std::invalid_argument   A model is not opened.
     *  @throw  std::runtime_error      An internal error.
     */
    double marginal(const std::string& y, const int t);
};

/**
 * Obtains the version number of the library.
 *  @return std::string     The version string.
 */
std::string version();

};

/** @} */

#endif/*__CRFSUITE_API_HPP__*/
