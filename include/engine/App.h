#pragma once

#include <memory>

struct GLFWwindow;
class Shader;
class DisplayTexture;
class Solver2D;

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
    void handleMouse();
    void frame(float timeSeconds);
    void drawUI();

    GLFWwindow* m_window = nullptr;

    std::unique_ptr<Shader>         m_display;
    std::unique_ptr<DisplayTexture> m_texture;
    std::unique_ptr<Solver2D>       m_solver;

    // An empty VAO. Core-profile OpenGL requires one bound for any draw call,
    // even though the fullscreen triangle is generated from gl_VertexID and
    // reads no vertex buffers at all.
    unsigned int m_vao = 0;

    // Simulation grid resolution. In Phase 0 this is only the size of the
    // texture CUDA scribbles on; from Phase 1 it becomes the actual grid.
    // 512x288 is 16:9, matching the default window, so grid cells are square
    // and a round splat renders round rather than as a wide ellipse.
    int m_simWidth  = 512;
    int m_simHeight = 288;

    float m_frameMs = 0.0f;

    // Mouse state, polled once per frame. The previous position is kept
    // because the velocity impulse is how far the cursor moved, which is only
    // knowable by comparing against the last frame.
    double m_mouseX = 0.0, m_mouseY = 0.0;
    double m_prevMouseX = 0.0, m_prevMouseY = 0.0;
    bool   m_mouseWasDown = false;

    // Splat parameters, driven by the ImGui sliders.
    float m_splatDye    = 1.0f;    // dye added at the centre of the blob
    float m_splatRadius = 14.0f;   // grid cells
    float m_splatForce  = 1.0f;    // impulse multiplier (no visible effect
                                   // until advection exists)
};
