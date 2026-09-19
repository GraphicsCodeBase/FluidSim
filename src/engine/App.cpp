#include "engine/App.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include "engine/DisplayTexture.h"
#include "engine/Shader.h"
#include "engine/Field.h"
#include "sim/testpattern.h"

#include <cstdio>

namespace {
void onGlfwError(int code, const char* description)
{
    std::fprintf(stderr, "[glfw] error %d: %s\n", code, description);
}
} // namespace

// Defined here rather than in the header because m_display/m_texture are
// unique_ptrs to forward-declared types.
App::App()  = default;
App::~App() = default;

bool App::init(int width, int height, const char* title)
{
    glfwSetErrorCallback(onGlfwError);
    if (!glfwInit()) return false;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    m_window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (!m_window) {
        std::fprintf(stderr, "[glfw] failed to create a 4.5 core context\n");
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(m_window);
    glfwSwapInterval(1); // vsync

    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
        std::fprintf(stderr, "[glad] failed to load OpenGL function pointers\n");
        return false;
    }

    std::printf("OpenGL  %s\n", glGetString(GL_VERSION));
    std::printf("GPU     %s\n", glGetString(GL_RENDERER));

    glCreateVertexArrays(1, &m_vao);

    m_display = std::make_unique<Shader>();
    if (!m_display->loadFromFiles(SHADER_DIR "/fullscreen.vert",
                                  SHADER_DIR "/display.frag")) {
        return false;
    }

    m_texture = std::make_unique<DisplayTexture>();
    if (!m_texture->create(m_simWidth, m_simHeight)) return false;

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(m_window, true);
    ImGui_ImplOpenGL3_Init("#version 450");

    return true;
}

void App::run()
{
    while (!glfwWindowShouldClose(m_window)) {
        glfwPollEvents();

        if (glfwGetKey(m_window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(m_window, GLFW_TRUE);

        const double start = glfwGetTime();

        frame(static_cast<float>(start));
        drawUI();

        // Measured before the buffer swap, so this is our own work rather
        // than time spent blocked waiting for vsync.
        m_frameMs = static_cast<float>((glfwGetTime() - start) * 1000.0);

        glfwSwapBuffers(m_window);
    }
}

void App::frame(float timeSeconds)
{
    // ---- CUDA owns the texture -------------------------------------------
    // In Phase 1 this block becomes the whole simulation step; for now it is
    // a single kernel painting a moving pattern.
    m_texture->map();
    launchTestPattern(m_texture->surface(), m_simWidth, m_simHeight, timeSeconds);
    m_texture->unmap();

    // ---- OpenGL owns the texture again -----------------------------------
    int fbWidth = 0, fbHeight = 0;
    glfwGetFramebufferSize(m_window, &fbWidth, &fbHeight);
    glViewport(0, 0, fbWidth, fbHeight);

    glClearColor(0.05f, 0.05f, 0.07f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    m_display->use();
    glBindTextureUnit(0, m_texture->glTexture());
    m_display->setInt("uField", 0);

    // Three vertices, no vertex buffer. See fullscreen.vert.
    glBindVertexArray(m_vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

void App::drawUI()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("Fluid Smoke");
    ImGui::TextUnformatted("Phase 0 - CUDA/OpenGL interop check");
    ImGui::Separator();
    ImGui::Text("Grid       %d x %d", m_simWidth, m_simHeight);
    ImGui::Text("Frame work %.2f ms", m_frameMs);
    ImGui::Text("Displayed  %.1f FPS", ImGui::GetIO().Framerate);
    ImGui::Separator();
    ImGui::TextWrapped(
        "If the pattern is animating, a CUDA kernel is writing directly into "
        "an OpenGL texture with no CPU round-trip. That is the only piece of "
        "novel plumbing in the project.");
    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void App::shutdown()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    m_texture.reset();
    m_display.reset();

    if (m_vao)    glDeleteVertexArrays(1, &m_vao);
    if (m_window) glfwDestroyWindow(m_window);
    glfwTerminate();
}
