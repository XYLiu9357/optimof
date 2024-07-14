/**main.cpp
 *
 * Invokes the RAC extractor with command line arguments
 */

#include <iostream>
#include <string>
#include <stdexcept>

#include "utils.hpp"

int main(int argc, char *argv[])
{
    // Malformed inputs
    if (argc != 3)
    {
        std::cerr << "RAC extractor expects 2 arguments, "
                  << argc - 1 << " were given\nExit with 1"
                  << std::endl;
        throw std::invalid_argument("main receives invalid arguments");
    }

    // Read inputs
    int argi = 0;
    std::string srcPath = argv[++argi];
    std::string destPath = argv[++argi];

    std::ifstream src = openFileRead(srcPath);
    std::ofstream dest = openFileWrite(destPath);

    src.close();
    dest.close();
}