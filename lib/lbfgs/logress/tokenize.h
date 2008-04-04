#ifndef __TOKENIZE_H__
#define __TOKENIZE_H__

template <class char_type>
class basic_tokenizer
{
protected:
    typedef basic_tokenizer<char_type> this_class;
    typedef typename std::basic_string<char_type> string_type;
    typedef typename string_type::const_iterator iterator_type;

    bool inl;
    string_type token;

    const string_type& line;
    iterator_type it;

public:
    basic_tokenizer(const string_type& _line) : inl(true), line(_line)
    {
        it = line.begin();
    }

    virtual ~basic_tokenizer()
    {
    }

    operator bool() const
    {
        return inl;
    }

    const string_type& operator*() const
    {
        return token;
    }

    const string_type* operator->() const
    {
        return &token;
    }

    this_class& next()
    {
        if (it != line.end()) {
            token.clear();
            for (;it != line.end();++it) {
                if (*it == '\t') {
                    ++it;
                    break;
                }
                token += *it;
            }
        } else {
            inl = false;
        }

        return *this;
    }
};

typedef basic_tokenizer<char> tokenizer;

#endif/*__TOKENIZE_H__*/
