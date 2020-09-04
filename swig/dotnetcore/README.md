## CRFsuite Microsoft .Net Core module via SWIG

##### HOW TO BUILD

````shell
~/crfsuite/swig$ cd dotnetcore
~/crfsuite/swig/dotnetcore$ swig -c++ -csharp -I../../include -o export_wrap.cpp ../export.i
~/crfsuite/swig/dotnetcore$ gcc -c -fpic -I../../include ../crfsuite.cpp export_wrap.cpp
~/crfsuite/swig/dotnetcore$ gcc -shared crfsuite.o export_wrap.o -o crfsuite.so
// compile .net project
~/crfsuite/swig/dotnetcore$ dotnet build ./dotnetcore.csproj
// move static lib to bin
~/crfsuite/swig/dotnetcore$ mv crfsuite.so ./bin/Debug/netcoreapp2.1
````



##### SAMPLE PROGRAMS
Get CRFsuite version
````shell
~/crfsuite/swig/dotnetcore$ dotnet ./bin/Debug/netcoreapp2.1/dotnetcore.dll
````
Refer to ~/crfsuite/swig/donetcore/Program.cs

Project Repository: https://github.com/Oceania2018/crfsuite

