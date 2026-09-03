#include <iostream>
#include <vector>
#include <string>
#include <cstring>

#if defined(_WIN32)
    #include <windows.h>
#else
    #include <dirent.h>
    #include <sys/stat.h>
#endif

#include "Augmentation.h"

// Cross-platform C++11 ReadDir implementation
void ReadDir(const char* dirPath, const char* extension, char**& files, int& nFiles) 
{
    std::vector<std::string> matchingFiles;

    // Standardize extension string (ensure leading dot)
    std::string extStr = extension;
    if (!extStr.empty() && extStr[0] != '.') {
        extStr = "." + extStr;
    }

#if defined(_WIN32)
    // Windows implementation
    std::string searchPath = dirPath;
    if (searchPath.back() != '\\' && searchPath.back() != '/') {
        searchPath += "\\";
    }
    std::string wildcardPath = searchPath + "*" + extStr;

    WIN32_FIND_DATAA findData;
    HANDLE hFind = FindFirstFileA(wildcardPath.c_str(), &findData);

    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            if (!(findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                matchingFiles.push_back(searchPath + findData.cFileName);
            }
        } while (FindNextFileA(hFind, &findData));
        FindClose(hFind);
    }
#else
    // Linux/POSIX implementation
    DIR* dir = opendir(dirPath);
    if (dir != nullptr) {
        struct dirent* entry;
        std::string baseDir = dirPath;
        if (baseDir.back() != '/') {
            baseDir += "/";
        }

        while ((entry = readdir(dir)) != nullptr) {
            std::string fileName = entry->d_name;
            
            // Check extension match
            if (fileName.length() >= extStr.length() &&
                fileName.compare(fileName.length() - extStr.length(), extStr.length(), extStr) == 0) {
                
                std::string fullPath = baseDir + fileName;
                
                // Verify it is a regular file
                struct stat st;
                if (stat(fullPath.c_str(), &st) == 0 && S_ISREG(st.st_mode)) {
                    matchingFiles.push_back(fullPath);
                }
            }
        }
        closedir(dir);
    }
#endif

    nFiles = static_cast<int>(matchingFiles.size());
    if (nFiles == 0) {
        files = nullptr;
        return;
    }

    // Allocate char** memory array
    files = new char*[nFiles];
    for (int i = 0; i < nFiles; ++i) {
        files[i] = new char[matchingFiles[i].length() + 1];
        std::strcpy(files[i], matchingFiles[i].c_str());
    }
}

// Memory cleanup helper function
void FreeDir(char** files, int nFiles) 
{
    if (!files) return;
    for (int i = 0; i < nFiles; ++i) {
        delete[] files[i];
    }
    delete[] files;
}

int main(int argc, char* argv[])
{
#if defined(_WIN32)
    const char* inputImageDir = "D:\\imagedir\\";
#else
    const char* inputImageDir = "/path/to/imagedir/";
#endif

    const char* extension = "jpg";
    char** in_files = nullptr;
    int in_nFiles = 0;

    // Pass in_files by reference (char**&) to capture the allocation
    ReadDir(inputImageDir, extension, in_files, in_nFiles);

    std::cout << "Found " << in_nFiles << " matching files.\n";

    for (int i = 0; i < in_nFiles; i++) {
        /* Writes the augmented image adding _aug1.jpg */
        writeAugmentedImage(in_files[i]);
    }

    // Free dynamically allocated memory
    FreeDir(in_files, in_nFiles);

    return 0;
}