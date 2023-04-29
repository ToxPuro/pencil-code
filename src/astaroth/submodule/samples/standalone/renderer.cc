/*
    Copyright (C) 2014-2022, Johannes Pekkila, Miikka Vaisala.

    This file is part of Astaroth.

    Astaroth is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Astaroth is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Astaroth.  If not, see <http://www.gnu.org/licenses/>.
*/

/**
 * @file
 * \brief Brief info.
 *
 * Detailed info.
 *
 */
#include "run.h"

#if AC_BUILD_RT_VISUALIZATION
#include <SDL.h>    // Note: using local version in src/3rdparty dir
#include <math.h>   // ceil
#include <string.h> // memcpy

#include "config_loader.h"
#include "errchk.h"
#include "math_utils.h"
#include "model/host_forcing.h"
#include "model/host_memory.h"
#include "model/host_timestep.h"
#include "model/model_reduce.h"
#include "model/model_rk3.h"
#include "timer_hires.h"

// NEED TO BE DEFINED HERE. IS NOT NOTICED BY compile_acc call.
#define LFORCING (1)
#define LSINK (0)

// Window
SDL_Renderer* renderer      = NULL;
static SDL_Window* window   = NULL;
static int window_width     = 800;
static int window_height    = 600;
static const int window_bpp = 32; // Bits per pixel

// Surfaces
SDL_Surface* surfaces[NUM_VTXBUF_HANDLES];
static int datasurface_width  = -1;
static int datasurface_height = -1;
static int k_slice            = 0;
static int k_slice_max        = 0;

// Colors
static SDL_Color color_bg      = (SDL_Color){30, 30, 35, 255};
static const int num_tiles     = NUM_VTXBUF_HANDLES + 1;
static const int tiles_per_row = 3;

/*
 * =============================================================================
 * Camera
 * =============================================================================
 */
/*
typedef struct {
   float x, y;
} float2;
*/
typedef struct {
    float x, y, w, h;
} vec4;

typedef struct {
    float2 pos;
    float scale;
} Camera;

static Camera camera = (Camera){(float2){.0f, .0f}, 1.f};

static inline vec4
project_ortho(const float2& pos, const float2& bbox, const float2& wdims)
{
    const vec4 rect = (vec4){camera.scale * (pos.x - camera.pos.x) + 0.5f * wdims.x,
                             camera.scale * (pos.y - camera.pos.y) + 0.5f * wdims.y,
                             camera.scale * bbox.x, camera.scale * bbox.y};

    return rect;
}

/*
 * =============================================================================
 * Renderer
 * =============================================================================
 */

static int
renderer_init(const int& mx, const int& my)
{
    // Init video
    SDL_InitSubSystem(SDL_INIT_VIDEO | SDL_INIT_EVENTS);

    // Setup window
    window = SDL_CreateWindow("Astaroth", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                              window_width, window_height, SDL_WINDOW_SHOWN);
    ERRCHK_ALWAYS(window);

    // Setup SDL renderer
    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    // SDL_SetWindowFullscreen(window, SDL_WINDOW_FULLSCREEN_DESKTOP);
    SDL_GetWindowSize(window, &window_width, &window_height);

    SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "1"); // Linear filtering

    datasurface_width  = mx;
    datasurface_height = my;
    // vec drawing uses the surface of the first component, no memory issues here
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i)
        surfaces[i] = SDL_CreateRGBSurfaceWithFormat(0, datasurface_width, datasurface_height,
                                                     window_bpp, SDL_PIXELFORMAT_RGBA8888);

    camera.pos   = (float2){.5f * tiles_per_row * datasurface_width - .5f * datasurface_width,
                            -.5f * (num_tiles / tiles_per_row) * datasurface_height +
                                .5f * datasurface_height};
    camera.scale = min(window_width / float(datasurface_width * tiles_per_row),
                       window_height / float(datasurface_height * (num_tiles / tiles_per_row)));

    SDL_RendererInfo renderer_info;
    SDL_GetRendererInfo(renderer, &renderer_info);
    printf("SDL renderer max texture dims: (%d, %d)\n", renderer_info.max_texture_width,
           renderer_info.max_texture_height);
    return 0;
}

static int
set_pixel(const int& i, const int& j, const uint32_t& color, SDL_Surface* surface)
{
    uint32_t* pixels           = (uint32_t*)surface->pixels;
    pixels[i + j * surface->w] = color;
    return 0;
}

static int
draw_vertex_buffer(const AcMesh& mesh, const VertexBufferHandle& vertex_buffer, const int& tile)
{
    const float xoffset = (tile % tiles_per_row) * datasurface_width;
    const float yoffset = -(tile / tiles_per_row) * datasurface_height;

    /*
    const float max = float(model_reduce_scal(mesh, RTYPE_MAX, vertex_buffer));
    const float min = float(model_reduce_scal(mesh, RTYPE_MIN, vertex_buffer));
    */
    const float max   = float(acReduceScal(RTYPE_MAX, vertex_buffer));
    const float min   = float(acReduceScal(RTYPE_MIN, vertex_buffer));
    const float range = fabsf(max - min);
    const float mid   = max - .5f * range;

    const int k = k_slice; // mesh.info.int_params[AC_mz] / 2;

    for (int j = 0; j < mesh.info.int_params[AC_my]; ++j) {
        for (int i = 0; i < mesh.info.int_params[AC_mx]; ++i) {
            ERRCHK(i < datasurface_width && j < datasurface_height);

            const int idx               = acVertexBufferIdx(i, j, k, mesh.info);
            const uint8_t shade         = (uint8_t)(255.f *
                                            (fabsf(float(mesh.vertex_buffer[vertex_buffer][idx]) -
                                                           mid)) /
                                            range);
            uint8_t color[4]            = {0, 0, 0, 255};
            color[tile % 3]             = shade;
            const uint32_t mapped_color = SDL_MapRGBA(surfaces[vertex_buffer]->format, color[0],
                                                      color[1], color[2], color[3]);
            set_pixel(i, j, mapped_color, surfaces[vertex_buffer]);
        }
    }

    const float2 pos   = (float2){xoffset, yoffset};
    const float2 bbox  = (float2){.5f * datasurface_width, .5f * datasurface_height};
    const float2 wsize = (float2){float(window_width), float(window_height)};
    const vec4 rectf   = project_ortho(pos, bbox, wsize);
    SDL_Rect rect      = (SDL_Rect){int(rectf.x - rectf.w), int(wsize.y - rectf.y - rectf.h),
                                    int(ceil(2.f * rectf.w)), int(ceil(2.f * rectf.h))};

    SDL_Texture* tex = SDL_CreateTextureFromSurface(renderer, surfaces[vertex_buffer]);
    SDL_RenderCopy(renderer, tex, NULL, &rect);
    SDL_DestroyTexture(tex);

    return 0;
}

static int
draw_vertex_buffer_vec(const AcMesh& mesh, const VertexBufferHandle& vertex_buffer_a,
                       const VertexBufferHandle& vertex_buffer_b,
                       const VertexBufferHandle& vertex_buffer_c, const int& tile)
{
    const float xoffset = (tile % tiles_per_row) * datasurface_width;
    const float yoffset = -(tile / tiles_per_row) * datasurface_height;

    /*
    const float maxx = float(
        max(model_reduce_scal(mesh, RTYPE_MAX, vertex_buffer_a),
            max(model_reduce_scal(mesh, RTYPE_MAX, vertex_buffer_b),
                model_reduce_scal(mesh, RTYPE_MAX, vertex_buffer_c))));
    const float minn = float(
        min(model_reduce_scal(mesh, RTYPE_MIN, vertex_buffer_a),
            min(model_reduce_scal(mesh, RTYPE_MIN, vertex_buffer_b),
                model_reduce_scal(mesh, RTYPE_MIN, vertex_buffer_c))));
    */
    const float maxx  = float(max(acReduceScal(RTYPE_MAX, vertex_buffer_a),
                                  max(acReduceScal(RTYPE_MAX, vertex_buffer_b),
                                      acReduceScal(RTYPE_MAX, vertex_buffer_c))));
    const float minn  = float(min(acReduceScal(RTYPE_MIN, vertex_buffer_a),
                                  min(acReduceScal(RTYPE_MIN, vertex_buffer_b),
                                      acReduceScal(RTYPE_MIN, vertex_buffer_c))));
    const float range = fabsf(maxx - minn);
    const float mid   = maxx - .5f * range;

    const int k = k_slice; // mesh.info.int_params[AC_mz] / 2;
    for (int j = 0; j < mesh.info.int_params[AC_my]; ++j) {
        for (int i = 0; i < mesh.info.int_params[AC_mx]; ++i) {
            ERRCHK(i < datasurface_width && j < datasurface_height);

            const int idx               = acVertexBufferIdx(i, j, k, mesh.info);
            const uint8_t r             = (uint8_t)(255.f *
                                        (fabsf(float(mesh.vertex_buffer[vertex_buffer_a][idx]) -
                                                           mid)) /
                                        range);
            const uint8_t g             = (uint8_t)(255.f *
                                        (fabsf(float(mesh.vertex_buffer[vertex_buffer_b][idx]) -
                                                           mid)) /
                                        range);
            const uint8_t b             = (uint8_t)(255.f *
                                        (fabsf(float(mesh.vertex_buffer[vertex_buffer_c][idx]) -
                                                           mid)) /
                                        range);
            const uint32_t mapped_color = SDL_MapRGBA(surfaces[vertex_buffer_a]->format, r, g, b,
                                                      255);
            set_pixel(i, j, mapped_color, surfaces[vertex_buffer_a]);
        }
    }

    const float2 pos   = (float2){xoffset, yoffset};
    const float2 bbox  = (float2){.5f * datasurface_width, .5f * datasurface_height};
    const float2 wsize = (float2){float(window_width), float(window_height)};
    const vec4 rectf   = project_ortho(pos, bbox, wsize);
    SDL_Rect rect      = (SDL_Rect){int(rectf.x - rectf.w), int(wsize.y - rectf.y - rectf.h),
                                    int(ceil(2.f * rectf.w)), int(ceil(2.f * rectf.h))};

    SDL_Texture* tex = SDL_CreateTextureFromSurface(renderer, surfaces[vertex_buffer_a]);
    SDL_RenderCopy(renderer, tex, NULL, &rect);
    SDL_DestroyTexture(tex);

    return 0;
}

static int
renderer_draw(const AcMesh& mesh)
{
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i)
        draw_vertex_buffer(mesh, VertexBufferHandle(i), i);
    draw_vertex_buffer_vec(mesh, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ, NUM_VTXBUF_HANDLES);

    // Drawing done, present
    SDL_RenderPresent(renderer);
    SDL_SetRenderDrawColor(renderer, color_bg.r, color_bg.g, color_bg.b, color_bg.a);
    SDL_RenderClear(renderer);

    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        const VertexBufferHandle vertex_buffer = VertexBufferHandle(i);
        /*
        printf("\t%s umax %e, min %e\n", vtxbuf_names[vertex_buffer],
               (double)model_reduce_scal(mesh, RTYPE_MAX, vertex_buffer),
               (double)model_reduce_scal(mesh, RTYPE_MIN, vertex_buffer));
        */
        printf("\t%s umax %e, min %e\n", vtxbuf_names[vertex_buffer],
               (double)acReduceScal(RTYPE_MAX, vertex_buffer),
               (double)acReduceScal(RTYPE_MIN, vertex_buffer));
    }
    printf("\n");

    return 0;
}

static int
renderer_quit(void)
{
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i)
        SDL_FreeSurface(surfaces[i]);

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);

    renderer = NULL;
    window   = NULL;

    SDL_Quit();
    return 0;
}

static int init_type = INIT_TYPE_GAUSSIAN_RADIAL_EXPL;
// static int init_type = INIT_TYPE_SIMPLE_CORE;

static bool
running(AcMesh* mesh)
{
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
        if (e.type == SDL_QUIT) {
            return false;
        }
        else if (e.type == SDL_KEYDOWN) {
            if (e.key.keysym.sym == SDLK_ESCAPE)
                return false;
            if (e.key.keysym.sym == SDLK_SPACE) {
                init_type = (init_type + 1) % NUM_INIT_TYPES;
                acmesh_init_to(InitType(init_type), mesh);
                acLoad(*mesh);
            }
            if (e.key.keysym.sym == SDLK_i) {
                k_slice = (k_slice + 1) % k_slice_max;
                printf("k_slice %d\n", k_slice);
            }
            if (e.key.keysym.sym == SDLK_k) {
                k_slice = (k_slice - 1 + k_slice_max) % k_slice_max;
                printf("k_slice %d\n", k_slice);
            }
        }
    }
    return true;
}

static void
check_input(const float& dt)
{
    /* Camera movement */
    const float camera_translate_rate = 1000.f / camera.scale;
    const float camera_scale_rate     = 1.0001f;
    const uint8_t* keystates          = (uint8_t*)SDL_GetKeyboardState(NULL);
    if (keystates[SDL_SCANCODE_UP])
        camera.pos.y += camera_translate_rate * dt;
    if (keystates[SDL_SCANCODE_DOWN])
        camera.pos.y -= camera_translate_rate * dt;
    if (keystates[SDL_SCANCODE_LEFT])
        camera.pos.x -= camera_translate_rate * dt;
    if (keystates[SDL_SCANCODE_RIGHT])
        camera.pos.x += camera_translate_rate * dt;
    if (keystates[SDL_SCANCODE_PAGEUP])
        camera.scale += camera.scale * camera_scale_rate * dt;
    if (keystates[SDL_SCANCODE_PAGEDOWN])
        camera.scale -= camera.scale * camera_scale_rate * dt;
    if (keystates[SDL_SCANCODE_COMMA])
        set_timescale(AcReal(.1));
    if (keystates[SDL_SCANCODE_PERIOD])
        set_timescale(AcReal(1.));
}

int
run_renderer(const char* config_path)
{
    /* Parse configs */
    AcMeshInfo mesh_info;
    load_config(config_path, &mesh_info);
    renderer_init(mesh_info.int_params[AC_mx], mesh_info.int_params[AC_my]);

    AcMesh* mesh = acmesh_create(mesh_info);
    acmesh_init_to(InitType(init_type), mesh);

#if LSINK
    vertex_buffer_set(VTXBUF_ACCRETION, 0.0, mesh);
#endif

    acInit(mesh_info);
    acLoad(*mesh);

    Timer frame_timer;
    timer_reset(&frame_timer);

    Timer wallclock;
    timer_reset(&wallclock);

    Timer io_timer;
    timer_reset(&io_timer);

    const float desired_frame_time = 1.f / 60.f;
    int steps                      = 0;
    k_slice                        = mesh->info.int_params[AC_mz] / 2;
    k_slice_max                    = mesh->info.int_params[AC_mz];
#if LSINK
    AcReal accreted_mass = 0.0;
#endif
    while (running(mesh)) {

        /* Input */
        check_input(timer_diff_nsec(io_timer) / 1e9f);
        timer_reset(&io_timer);

/* Step the simulation */
#if LSINK
        const AcReal sum_mass = acReduceScal(RTYPE_SUM, VTXBUF_ACCRETION);
        accreted_mass         = accreted_mass + sum_mass;
        AcReal sink_mass      = mesh_info.real_params[AC_M_sink_init] + accreted_mass;
        printf("sink mass is: %e \n", sink_mass);
        printf("accreted mass is: %e \n", accreted_mass);
        acLoadDeviceConstant(AC_M_sink, sink_mass);
        vertex_buffer_set(VTXBUF_ACCRETION, 0.0, mesh);
#endif

#if LFORCING
        const ForcingParams forcing_params = generateForcingParams(mesh->info);
        loadForcingParamsToDevice(forcing_params);
#endif

#if 1
        const AcReal umax = acReduceVec(RTYPE_MAX, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ);
        const AcReal dt   = host_timestep(umax, mesh_info);

        acIntegrate(dt);
#else
        ModelMesh* model_mesh = modelmesh_create(mesh->info);
        const AcReal umax     = AcReal(
                model_reduce_vec(*model_mesh, RTYPE_MAX, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ));
        const AcReal dt = host_timestep(umax, mesh_info);
        acmesh_to_modelmesh(*mesh, model_mesh);
        model_rk3(dt, model_mesh);
        modelmesh_to_acmesh(*model_mesh, mesh);
        modelmesh_destroy(model_mesh);
        acLoad(*mesh); // Just a quick hack s.t. we do not have to add an
                       // additional if to the render part
#endif

        ++steps;

        /* Render */
        const float timer_diff_sec = timer_diff_nsec(frame_timer) / 1e9f;
        if (timer_diff_sec >= desired_frame_time) {
            const int num_vertices = mesh->info.int_params[AC_mxy];
            const int3 dst         = (int3){0, 0, k_slice};
            acBoundcondStep();
            // acStore(mesh);
            acStoreWithOffset(dst, num_vertices, mesh);
            acSynchronizeStream(STREAM_ALL);
            renderer_draw(*mesh); // Bottleneck is here
            printf("Step #%d, dt: %f\n", steps, double(dt));
            timer_reset(&frame_timer);
        }
    }
    printf("Wallclock time %f s\n", double(timer_diff_nsec(wallclock) / 1e9f));

    acQuit();
    acmesh_destroy(mesh);

    renderer_quit();

    return 0;
}
#else // BUILD_RT_VISUALIZATION == 0
#include "errchk.h"
int
run_renderer(const char* /*config_path*/)
{
    WARNING("Real-time visualization module not built. Set BUILD_RT_VISUALIZATION=ON with cmake.");
    return 1;
}
#endif // BUILD_RT_VISUALIZATION
