#### Installation instructions for Mac OS X El Capitan with homebrew clang-omp
```Shell
> brew install autoconf
> brew install automake
> brew install libtool
> brew install clang-omp
> ./autogen.sh
> ./configure CC=clang-omp
> make
> make install
```
