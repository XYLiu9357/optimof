/**utils.hpp
 *
 * Utility functions use in RAC extractor.
 */

#include <string>
#include <fstream>

#include "utils.hpp"

std::ifstream openFileRead(const std::string &filePath)
{
    std::ifstream fileHandle(filePath);
    if (fileHandle.fail())
        throw std::runtime_error("Could not open file" + filePath);

    return fileHandle;
}

std::ofstream openFileWrite(const std::string &filePath)
{
    std::ofstream fileHandle(filePath);
    if (fileHandle.fail())
        throw std::runtime_error("Could not open file" + filePath);

    return fileHandle;
}