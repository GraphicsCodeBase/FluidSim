#include "engine/App.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include "engine/DisplayTexture.h"
#include "engine/Shader.h"
#include "sim/Solver2D.h"

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

    // The simulation. Created after the texture because both allocate GPU
    // memory and it keeps the startup log in a sensible order.
    m_solver = std::make_unique<Solver2D>();
    m_solver->create(m_simWidth, m_simHeight);

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

        handleMouse();
        frame(static_cast<float>(start));
        drawUI();

        // Measured before the buffer swap, so this is our own work rather
        // than time spent blocked waiting for vsync.
        m_frameMs = static_cast<float>((glfwGetTime() - start) * 1000.0);

        glfwSwapBuffers(m_window);
    }
}

void App::handleMouse()
{
    // Let ImGui have the mouse when the cursor is over a panel - otherwise
    // dragging a slider would also paint smoke behind it.
    if (ImGui::GetIO().WantCaptureMouse) {
        m_mouseWasDown = false;
        return;
    }

    glfwGetCursorPos(m_window, &m_mouseX, &m_mouseY);
    const bool down =
        glfwGetMouseButton(m_window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;

    int windowW = 0, windowH = 0;
    glfwGetWindowSize(m_window, &windowW, &windowH);

    if (down && windowW > 0 && windowH > 0) {
        // Window pixels -> grid cells.
        //
        // GLFW measures y DOWNWARD from the top of the window. Our grid
        // measures y UPWARD from the bottom, so that buoyancy will simply be
        // +y in Phase 3 with no sign flips. Hence the (1.0 - ...) below.
        //
        //   cursor at window top    -> mouseY = 0    -> gy = m_simHeight
        //   cursor at window bottom -> mouseY = H    -> gy = 0
        const float gx = static_cast<float>(m_mouseX / windowW) * m_simWidth;
        const float gy = static_cast<float>(1.0 - m_mouseY / windowH) * m_simHeight;

        // The impulse is how far the cursor travelled since the last frame,
        // expressed in grid cells. A stationary click therefore adds dye but
        // no motion, which is exactly what we want.
        float impulseX = 0.0f;
        float impulseY = 0.0f;

        // Only if the button was ALREADY down last frame. On the first frame
        // of a click the previous position is wherever the cursor happened to
        // be beforehand, which would inject one enormous bogus impulse.
        if (m_mouseWasDown) {
            impulseX =  static_cast<float>((m_mouseX - m_prevMouseX) / windowW) * m_simWidth;
            impulseY = -static_cast<float>((m_mouseY - m_prevMouseY) / windowH) * m_simHeight;
            //         ^ same y flip, for the same reason: dragging the cursor
            //           down should push the fluid down, i.e. negative grid y.
        }

        m_solver->splat(gx, gy,
                        impulseX * m_splatForce, impulseY * m_splatForce,
                        m_splatDye, m_splatRadius);
    }

    m_prevMouseX   = m_mouseX;
    m_prevMouseY   = m_mouseY;
    m_mouseWasDown = down;
}

void App::frame(float timeSeconds)
{
    // ---- CUDA owns the texture -------------------------------------------
    // The solver copies its current dye field into the display texture. As
    // more operators arrive this block grows into the whole simulation step.
    m_texture->map();
    m_solver->renderTo(m_texture->surface());

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
    ImGui::TextUnformatted("Phase 1 - step 4: painting");
    ImGui::Separator();
    ImGui::Text("Grid       %d x %d", m_simWidth, m_simHeight);
    ImGui::Text("Frame work %.2f ms", m_frameMs);
    ImGui::Text("Displayed  %.1f FPS", ImGui::GetIO().Framerate);

    ImGui::Separator();
    ImGui::TextUnformatted("Splat");
    ImGui::SliderFloat("dye",    &m_splatDye,    0.0f,  2.0f);
    ImGui::SliderFloat("radius", &m_splatRadius, 2.0f, 60.0f);
    ImGui::SliderFloat("force",  &m_splatForce,  0.0f, 10.0f);

    ImGui::Separator();
    ImGui::TextWrapped(
        "Drag with the left mouse button to paint. The dye will not move - "
        "there is no advection yet, so the velocity impulse being injected "
        "has nothing acting on it. That is the next operator.");
    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void App::shutdown()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    m_solver.reset();
    m_texture.reset();
    m_display.reset();

    if (m_vao)    glDeleteVertexArrays(1, &m_vao);
    if (m_window) glfwDestroyWindow(m_window);
    glfwTerminate();
}
