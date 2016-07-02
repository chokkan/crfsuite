#### Installation instructions for Mac OS X El Capitan with homebrew gcc-6 OpenMP
```Shell
> brew install autoconf
> brew install automake
> brew install libtool
> brew install gcc --without-multilib
> ./autogen.sh
> ./configure CC=gcc-6
> make
> make install
```
