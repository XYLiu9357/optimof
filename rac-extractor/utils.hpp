/**utils.hpp
 *
 * Header interface for utility functions use in RAC extractor.
 */

#ifndef UTILS
#define UTILS

#include <string>
#include <fstream>

/**
 * I/O wrappers
 *
 * Wrappers support exit on failure with warnings. User should close
 * the streams manually once they are no longer needed.
 */
std::ifstream openFileRead(const std::string &filePath);
std::ofstream openFileWrite(const std::string &filePath);

#endif /*UTILS*/