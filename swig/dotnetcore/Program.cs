using System;

namespace dotnetcore
{
    class Program
    {
        static void Main(string[] args)
        {
            var crfsuite = new crfsuite();
            Console.WriteLine(crfsuite.version());
        }
    }
}
