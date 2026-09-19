#include "engine/Shader.h"

#include <glad/glad.h>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <vector>

namespace {

bool readFile(const std::string& path, std::string& out)
{
    std::ifstream file(path);
    if (!file) {
        std::fprintf(stderr, "[shader] cannot open %s\n", path.c_str());
        return false;
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    out = ss.str();
    return true;
}

GLuint compile(GLenum stage, const std::string& source, const std::string& label)
{
    GLuint shader = glCreateShader(stage);
    const char* src = source.c_str();
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    GLint ok = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        GLint len = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);
        std::vector<char> log(len > 1 ? len : 1);
        glGetShaderInfoLog(shader, len, nullptr, log.data());
        std::fprintf(stderr, "[shader] %s failed to compile:\n%s\n", label.c_str(), log.data());
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

} // namespace

Shader::~Shader()
{
    if (m_program) glDeleteProgram(m_program);
}

bool Shader::loadFromFiles(const std::string& vertexPath, const std::string& fragmentPath)
{
    std::string vertSrc, fragSrc;
    if (!readFile(vertexPath, vertSrc))   return false;
    if (!readFile(fragmentPath, fragSrc)) return false;

    GLuint vert = compile(GL_VERTEX_SHADER,   vertSrc, vertexPath);
    GLuint frag = compile(GL_FRAGMENT_SHADER, fragSrc, fragmentPath);
    if (!vert || !frag) {
        if (vert) glDeleteShader(vert);
        if (frag) glDeleteShader(frag);
        return false;
    }

    m_program = glCreateProgram();
    glAttachShader(m_program, vert);
    glAttachShader(m_program, frag);
    glLinkProgram(m_program);

    GLint ok = GL_FALSE;
    glGetProgramiv(m_program, GL_LINK_STATUS, &ok);
    if (!ok) {
        GLint len = 0;
        glGetProgramiv(m_program, GL_INFO_LOG_LENGTH, &len);
        std::vector<char> log(len > 1 ? len : 1);
        glGetProgramInfoLog(m_program, len, nullptr, log.data());
        std::fprintf(stderr, "[shader] link failed:\n%s\n", log.data());
        glDeleteProgram(m_program);
        m_program = 0;
    }

    glDeleteShader(vert);
    glDeleteShader(frag);
    return m_program != 0;
}

void Shader::use() const
{
    glUseProgram(m_program);
}

void Shader::setInt(const char* name, int value) const
{
    glUniform1i(glGetUniformLocation(m_program, name), value);
}

void Shader::setFloat(const char* name, float value) const
{
    glUniform1f(glGetUniformLocation(m_program, name), value);
}
