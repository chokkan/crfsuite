%module(directors="1") crfsuite

%{
#include "crfsuite_api.hpp"
%}

%include "std_string.i"
%include "std_vector.i"
%include "exception.i"

%rename(Item) std::vector<CRFSuite::Attribute>;
%rename(ItemSequence) std::vector<CRFSuite::Item>;
%rename(StringList) std::vector<std::string>;

%feature("director") Trainer;

%exception {
    try {
        $action
    } catch(const std::invalid_argument& e) {
        SWIG_exception(SWIG_IOError, e.what());
    } catch(const std::runtime_error& e) {
        SWIG_exception(SWIG_RuntimeError, e.what());
    } catch (const std::exception& e) {
        SWIG_exception(SWIG_RuntimeError, e.what());
    } catch(...) {
        SWIG_exception(SWIG_RuntimeError,"Unknown exception");
    }
}

%include "crfsuite_api.hpp"

