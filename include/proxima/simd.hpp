#pragma once

// Single switch for the NEON fast paths. PROXIMA_FORCE_SCALAR (a CMake
// option) compiles the scalar fallbacks even on Apple Silicon, those
// branches otherwise never get built on the primary dev machine, and
// untested fallback code is where bit-rot lives. CI builds both flavors.
#if defined(__ARM_NEON) && !defined(PROXIMA_FORCE_SCALAR)
#define PROXIMA_USE_NEON 1
#include <arm_neon.h>
#else
#define PROXIMA_USE_NEON 0
#endif
