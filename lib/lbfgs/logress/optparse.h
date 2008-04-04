/*
 *      An event-driven parser for command-line arguments.
 *  
 *      Copyright (c) 2004-2005 by N.Okazaki
 *
 * This software is provided 'as-is', without any express or implied
 * warranty.  In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions (known as zlib license):
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgment in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 *
 * Naoaki Okazaki <okazaki at chokkan dot org>
 *
 */

/* $Id$ */

/*
 * Class 'optparse' implements a parser for GNU-style command-line arguments.
 * Inherit this class to define your own option variables and to implement an
 * option handler with macros, BEGIN_OPTION_MAP, ON_OPTION(_WITH_ARG), and
 * END_OPTION_MAP. Consult the sample program attached at the bottom of this
 * source code.
 *
 * This code was comfirmed to be compiled with MCVC++ 2003 and gcc 3.3.
 * Define _BUILD_NCL_SAMPLE if you want to build a sample program.
 *  $ g++ -D_BUILD_NCL_SAMPLE -xc++ optparse.h
 */

#ifndef __NCL_OPTPRASE_H__
#define __NCL_OPTPRASE_H__

#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>


#ifdef  USE_NCL_NAMESPACE
namespace ncl {
#endif/*USE_NCL_NAMESPACE*/


/**
 * An event-driven parser for command-line arguments.
 *  @author Naoaki Okazaki
 */
class optparse {
public:
    /**
     * Exception class for unrecognized options.
     */
    class unrecognized_option : public std::invalid_argument {
    public:
        unrecognized_option(char shortopt)
            : std::invalid_argument(std::string("-") + shortopt) {}
        unrecognized_option(const std::string& longopt)
            : std::invalid_argument(std::string("--") + longopt) {}
    };
    /**
     * Exception class for invalid values.
     */
    class invalid_value : public std::invalid_argument {
    public:
        invalid_value(const std::string& message)
            : std::invalid_argument(message) {}
    };

public:
    /** Construct. */
    optparse() {}
    /** Destruct. */
    virtual ~optparse() {}

    /**
     * Parse options.
     *  @param  argv        array of null-terminated strings to be parsed
     *  @param  num_argv    specifies the number, in strings, of the array
     *  @return             the number of used arguments
     *  @throws             optparse_exception
     */
    int parse(char * const argv[], int num_argv, int arg_start = 1)
    {
        int i;
        for (i = arg_start;i < num_argv;++i) {
            const char *token = argv[i];
            if (*token++ == '-') {
                const char *next_token = (i+1 < num_argv) ? argv[i+1] : "";
                if (!*token) {
                    break;  // only '-' was found.
                } else if (*token == '-') {
                    const char *arg = std::strchr(++token, '=');
                    if (arg) {
                        arg++;
                    } else {
                        arg = next_token;
                    }
                    int ret = handle_option(0, token, arg);
                    if (ret < 0) {
                        throw unrecognized_option(token);
                    }
                    if (arg == next_token) {
                        i += ret;
                    }
                } else {
                    char c;
                    while ((c = *token++) != '\0') {
                        const char *arg = *token ? token : next_token;
                        int ret = handle_option(c, token, arg);
                        if (ret < 0) {
                            throw unrecognized_option(c);
                        }
                        if (ret > 0) {
                            if (arg == token) {
                                token = "";
                            } else {
                                i++;
                            }
                        }
                    } // while
                } // else (*token == '-') 
            } else {
                break;  // a non-option argument was fonud.
            } 
        } // for (i)

        return i;
    }

protected:
    /**
     * Option handler
     *  This function should be overridden by inheritance class.
     *  @param  c           short option character, 0 for long option
     *  @param  longname    long option name
     *  @param  arg         an argument for the option
     *  @return             0 (success);
                            1 (success with use of an argument);
                            -1 (failed, unrecognized option)
     *  @throws             option_parser_exception
     */
    virtual int handle_option(char c, const char *longname, const char *arg)
    {
        return 0;
    }

    int __optstrcmp(const char *option, const char *longname)
    {
        const char *p = std::strchr(option, '=');
        return p ?
            std::strncmp(option, longname, p-option) :
            std::strcmp(option, longname);
    }
};


/** The begin of inline option map. */
#define BEGIN_OPTION_MAP_INLINE() \
    virtual int handle_option(char __c, const char *__longname, const char *arg) \
    { \
        int used_args = 0; \
        if (0) { \

/** Define of option map. */
#define DEFINE_OPTION_MAP() \
    virtual int handle_option(char __c, const char *__longname, const char *arg);

/** Begin of option map implimentation. */
#define BEGIN_OPTION_MAP(_Class) \
    int _Class::handle_option(char __c, const char *__longname, const char *arg) \
    { \
        int used_args = 0; \
        if (0) { \

/** An entry of option map */
#define ON_OPTION(test) \
            return used_args; \
        } else if (test) { \
            used_args = 0; \

#define ON_OPTION_WITH_ARG(test) \
            return used_args; \
        } else if (test) { \
            used_args = 1; \

/** The end of option map implementation */
#define END_OPTION_MAP() \
            return used_args; \
        } \
        return -1; \
    } \

/** A predicator for short options */
#define SHORTOPT(x)     (__c == x)
/** A predicator for long options */
#define LONGOPT(x)      (!__c && __optstrcmp(__longname, x) == 0)


#ifdef  USE_NCL_NAMESPACE
};
#endif/*USE_NCL_NAMESPACE*/






#ifdef  _BUILD_NCL_SAMPLE

#include <cstdio>
#include <iostream>

/**
 * A class to store parameters specified by command-line arguments
 */
class option : public optparse {
public:
    int bytes;
    int lines;
    bool quiet;

    option() : bytes(0), lines(0), quiet(false) {}

    BEGIN_OPTION_MAP_INLINE()
        ON_OPTION(SHORTOPT('b') || LONGOPT("bytes"))
            bytes = std::atoi(arg);
            used_args = 1;  // Notify the parser of a consumption of argument.

        ON_OPTION_WITH_ARG(SHORTOPT('l') || LONGOPT("lines"))
            lines = std::atoi(arg);
            // no need of the notification: used_args variable will be set to 1.

        ON_OPTION(SHORTOPT('q') || LONGOPT("quiet") || LONGOPT("silent"))
            quiet = true;

    END_OPTION_MAP()
};

int main(int argc, char *argv[])
{
    try {
        option opt;
        int argused = opt.parse(&argv[1], argc-1); // Skip argv[0].

        std::cout << "used argv: " << argused << std::endl;
        std::cout << "bytes: " << opt.bytes << std::endl;
        std::cout << "lines: " << opt.lines << std::endl;
        std::cout << "quiet: " << opt.quiet << std::endl;
    } catch (const optparse::unrecognized_option& e) {
        std::cout << "unrecognized option: " << e.what() << std::endl;
        return 1;
    } catch (const optparse::invalid_value& e) {
        std::cout << "invalid value: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

#endif/*_BUILD_NCL_SAMPLE*/


#endif/*__NCL_OPTPRASE_H__*/
