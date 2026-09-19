#pragma once

#include <memory>

struct GLFWwindow;
class Shader;
class DisplayTexture;

// ---------------------------------------------------------------------------
// Layer 1 - Foundation.
//
// Window, OpenGL context, ImGui, and the frame loop. There is no physics in
// here and there never will be; the simulation is a separate layer that this
// one merely calls into and displays.
// ---------------------------------------------------------------------------
class App
{
public:
    App();
    ~App();

    bool init(int width, int height, const char* title);
    void run();
    void shutdown();

private:
    void frame(float timeSeconds);
    void drawUI();

    GLFWwindow* m_window = nullptr;

    std::unique_ptr<Shader>         m_display;
    std::unique_ptr<DisplayTexture> m_texture;

    // An empty VAO. Core-profile OpenGL requires one bound for any draw call,
    // even though the fullscreen triangle is generated from gl_VertexID and
    // reads no vertex buffers at all.
    unsigned int m_vao = 0;

    // Simulation grid resolution. In Phase 0 this is only the size of the
    // texture CUDA scribbles on; from Phase 1 it becomes the actual grid.
    int m_simWidth  = 512;
    int m_simHeight = 512;

    float m_frameMs = 0.0f;
};
