#include "engine/App.h"

#include <cstdio>

int main()
{
    App app;

    if (!app.init(1280, 720, "Fluid Smoke - Phase 0")) {
        std::fprintf(stderr, "Initialisation failed.\n");
        return 1;
    }

    app.run();
    app.shutdown();
    return 0;
}
