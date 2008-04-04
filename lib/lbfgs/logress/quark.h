/*
 *      Quark (association between a string and an integer ID) class.
 *
 *      Copyright (c) 2004-2008 Naoaki Okazaki
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
Quark allocates an unique ID for each string and holds the association so that
we can obtain the string value associated with an ID and/or the ID value
associated with a string value. Since string matching is slower than integer
comparison, it's a common technique for speed/memory optimization that an
application converts all string values into integer identifiers, does some
process with the integer values, and then restores the string values from
them.

This code was comfirmed to be compiled with MCVC++ 2003 and gcc 3.4.4
Define _BUILD_NCL_SAMPLE if you want to build a sample program.
    $ g++ -D_BUILD_NCL_SAMPLE -xc++ quark.h
*/


#ifndef __QUARK_H__
#define __QUARK_H__

#include <vector>
#include <string>


#ifdef  USE_NCL_NAMESPACE
namespace ncl {
#endif/*USE_NCL_NAMESPACE*/


#if     defined(_MSC_VER)

#include <hash_map>
#define HashMap     stdext::hash_map

#elif   defined(__GNUC__)

#include <locale>
#include <ext/hash_map>
#define HashMap     __gnu_cxx::hash_map

namespace __gnu_cxx
{
    template<> struct hash< std::string >
    {
        // We define our hash function here.
        size_t operator()( const std::string& x ) const
        {
            std::locale loc;
            return std::use_facet< std::collate<char> >(loc).hash(x.c_str(), x.c_str() + x.length());
        }
    };
}

#else

#error  "Define a hash_map class for your compiler. "

#endif


/**
 * The basic class of quark class.
 *  @param  string_t                String class name to be used.
 *  @param  qid_t                   ID class name to be used.
 *  @author Naoaki Okazaki
 */
template <class string_t, class qid_t>
class basic_quark {
protected:
    typedef HashMap<string_t, qid_t> StringToId;
    typedef std::vector<string_t> IdToString;

    StringToId m_string_to_id;
    IdToString m_id_to_string;

public:
    const qid_t npos;

public:
    /**
     * Construct.
     */
    basic_quark() : npos((qid_t)(-1))
    {
        clear();
    }

    /**
     * Destruct.
     */
    virtual ~basic_quark()
    {
    }

    /**
     * Map a string to its associated ID.
     *  If string-to-integer association does not exist, allocate a new ID.
     *  @param  str                 String value.
     *  @return                     Associated ID for the string value.
     */
    qid_t operator[](const string_t& str)
    {
        typename StringToId::const_iterator it = m_string_to_id.find(str);
        if (it != m_string_to_id.end()) {
            return it->second;
        } else {
            qid_t newid = (qid_t)m_id_to_string.size();
            m_id_to_string.push_back(str);
            m_string_to_id.insert(std::pair<string_t, qid_t>(str, newid));
            return newid;
        }
    }

    /**
     * Convert ID value into the associated string value.
     *  @param  qid                 ID.
     *  @param  def                 Default value if the ID was out of range.
     *  @return                     String value associated with the ID.
     */
    const string_t& to_string(const qid_t& qid, const string_t& def = "") const
    {
        if (0 <= qid && qid < m_id_to_string.size()) {
            return m_id_to_string[qid];
        } else {
            return def;
        }
    }

    /**
     * Convert string value into the associated ID value.
     *  @param  str                 String value.
     *  @return                     ID if any, otherwise 0.
     */
    qid_t to_id(const string_t& str) const
    {
        typename StringToId::const_iterator it = m_string_to_id.find(str);
        if (it != m_string_to_id.end()) {
            return it->second;
        } else {
            return npos;
        }

    }

    void clear()
    {
        m_string_to_id.clear();
        m_id_to_string.clear();
    }

    /**
     * Get the number of string-to-id associations.
     *  @return                     The number of association.
     */
    size_t size() const
    {
        return m_id_to_string.size();
    }
};

/**
 * Specialized quark class with std::string to int association.
 */
typedef basic_quark<std::string, int> quark;


#ifdef  USE_NCL_NAMESPACE
};
#endif/*USE_NCL_NAMESPACE*/





#ifdef  _BUILD_NCL_SAMPLE

#include <iostream>

int main(int argc, char *argv[])
{
    quark q;

    std::cout << q["you"] << std::endl;                             // 1
    std::cout << q["your"] << std::endl;                            // 2
    std::cout << q["you"] << std::endl;                             // 1
    std::cout << q["yours"] << std::endl;                           // 3

    std::cout << 1 << ": " << q.from_id(1) << std::endl;            // 1: you
    std::cout << 2 << ": " << q.from_id(2) << std::endl;            // 2: your
    std::cout << 3 << ": " << q.from_id(3) << std::endl;            // 3: yours
    std::cout << 4 << ": " << q.from_id(4) << std::endl;            // 4:

    std::cout << "you: " << q.from_string("you") << std::endl;      // you: 1
    std::cout << "yours: " << q.from_string("yours") << std::endl;  // yours: 3
    std::cout << "I: " << q.from_string("I") << std::endl;          // I: 0

    return 0;
}

#endif/*_BUILD_NCL_SAMPLE*/

#endif/*__QUARK_H__*/
