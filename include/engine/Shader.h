#pragma once

#include <string>

// Minimal GLSL program loader: read two files, compile, link, report errors.
class Shader
{
public:
    ~Shader();

    bool loadFromFiles(const std::string& vertexPath, const std::string& fragmentPath);

    void use() const;
    void setInt(const char* name, int value) const;
    void setFloat(const char* name, float value) const;

    unsigned int id() const { return m_program; }

private:
    unsigned int m_program = 0;
};
