


#if 1
//
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
//
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>
#include <array>
#include <iostream>
#include <type_traits>
#include <thread>
#include <atomic>
//
#include "rayrun.hpp"
#include "vec3.h"
#include "simd_vec.h"

#include <immintrin.h>

//
BOOL APIENTRY DllMain(HMODULE hModule,
    DWORD  ul_reason_for_call,
    LPVOID lpReserved
)
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}

// #define MYTHREADING

bool neverUseOpenMP()
{
#ifdef MYTHREADING
    return true;
#else
    return false;
#endif
}

static constexpr int PackedTriangle = 16;

namespace impl
{
    std::vector<float> vertices;
    std::vector<float> normals;
    struct Triangle
    {
        math::Float3 v[3];
        math::Float3 n[3];
    };
    std::vector<Triangle> triangles;

    // 512 = 32 * 16
    // 16 triangles単位
    struct alignas(64) SIMDTrianglePack
    {
        __m512 x[3];
        __m512 y[3];
        __m512 z[3];
        uint32_t triangle_offset;
    };

    std::vector<SIMDTrianglePack> simd_triangles;


    alignas(64) float t0_f[PackedTriangle];
    alignas(64) float t1_f[PackedTriangle];
    alignas(64) float zero_data[PackedTriangle];
    alignas(64) float one_data[PackedTriangle];
    alignas(64) float eps_data[PackedTriangle];

    __m512 zero;
    __m512 one;
    __m512 keps;

    void init()
    {
        constexpr float eps = 1e-5f;
        for (int i = 0; i < PackedTriangle; ++i)
        {
            t0_f[i] = -eps;
            t1_f[i] = 1 + eps;
            zero_data[i] = 0;
            one_data[i] = 1;
            eps_data[i] = 1e-6f;
        }
        zero = _mm512_load_ps(zero_data);
        one = _mm512_load_ps(one_data);
        keps = _mm512_load_ps(eps_data);
    }

    struct Hitpoint
    {
        bool hit = false;
        float distance = std::numeric_limits<float>::infinity();
        float b1, b2;
        int index;
    };

    void ray_vs_triangle(__m512& current_distance, __m512 near_t, __m512 far_t, __m512 org[3], __m512 dir[3], const SIMDTrianglePack& s, Hitpoint* hitpoint)
    {
        alignas(64) float t_f[PackedTriangle];
        alignas(64) float b1_f[PackedTriangle];
        alignas(64) float b2_f[PackedTriangle];

        auto t0 = _mm512_load_ps(t0_f);
        auto t1 = _mm512_load_ps(t1_f);

        auto e1_x = s.x[1] - s.x[0];
        auto e1_y = s.y[1] - s.y[0];
        auto e1_z = s.z[1] - s.z[0];

        auto e2_x = s.x[2] - s.x[0];
        auto e2_y = s.y[2] - s.y[0];
        auto e2_z = s.z[2] - s.z[0];

        auto s1_x = dir[1] * e2_z - dir[2] * e2_y;
        auto s1_y = dir[2] * e2_x - dir[0] * e2_z;
        auto s1_z = dir[0] * e2_y - dir[1] * e2_x;

        auto divisor = s1_x * e1_x + s1_y * e1_y + s1_z * e1_z;
        auto no_hit = divisor == zero;
        auto invDivisor = one / divisor;

        auto d_x = org[0] - s.x[0];
        auto d_y = org[1] - s.y[0];
        auto d_z = org[2] - s.z[0];

        auto b1 = (d_x * s1_x + d_y * s1_y + d_z * s1_z) * invDivisor;
        no_hit = or(no_hit, or(b1 < t0, b1 > t1));

        auto s2_x = d_y * e1_z - d_z * e1_y;
        auto s2_y = d_z * e1_x - d_x * e1_z;
        auto s2_z = d_x * e1_y - d_y * e1_x;

        auto b2 = (dir[0] * s2_x + dir[1] * s2_y + dir[2] * s2_z) * invDivisor;
        no_hit = or(no_hit, or(b2 < t0, (b1 + b2) > t1));

        auto t = (e2_x * s2_x + e2_y * s2_y + e2_z * s2_z) * invDivisor;
        no_hit = or(no_hit, t < keps);

#if 1
        auto hit_flag = and(not(no_hit), and(and(near_t < t, t < far_t), t < current_distance));
        int hit_flag_int = hit_flag;
        if (hit_flag_int == 0)
            return;

        _mm512_store_ps(t_f, t);
        _mm512_store_ps(b1_f, b1);
        _mm512_store_ps(b2_f, b2);

        int k = 0;
        for (int i = 0; i < PackedTriangle; ++i) {
            if ((hit_flag_int & (1 << i)) && t_f[i] < hitpoint->distance)
            {
                hitpoint->hit = true;
                hitpoint->index = s.triangle_offset + i;
                hitpoint->distance = t_f[i];
                hitpoint->b1 = b1_f[i];
                hitpoint->b2 = b2_f[i];
                k = 1;
            }
        }

        if (k)
        {
            alignas(64) float dist[PackedTriangle];
            std::fill(dist, dist + PackedTriangle, hitpoint->distance);
            current_distance = _mm512_load_ps(dist);
        }
#endif
#if 0
        int nohitmask = no_hit;

        _mm512_store_ps(t_f, t);
        _mm512_store_ps(b1_f, b1);
        _mm512_store_ps(b2_f, b2);

        for (int i = 0; i < PackedTriangle; ++i) {
            auto current_t = t_f[i];
            if ((nohitmask & (1 << i)) == 0 && hitpoint->distance > current_t) {
                if (near_t <= current_t && current_t <= far_t)
                {
                    hitpoint->hit = true;
                    hitpoint->index = s.triangle_offset + i;
                    hitpoint->distance = current_t;
                    hitpoint->b1 = b1_f[i];
                    hitpoint->b2 = b2_f[i];
                }
            }
        }
#endif
    }


    static constexpr int NUM_THREAD = 16;

    void set_thread_group(uint32_t thread_id)
    {
        auto group = thread_id % 2; // 0, 1で、プロセッサグループを指定する。っぽい。環境依存。
        GROUP_AFFINITY mask;
        if (GetNumaNodeProcessorMaskEx(group, &mask))
            SetThreadGroupAffinity(GetCurrentThread(), &mask, nullptr);
    }

    std::vector<std::thread> thraed_pool;

    void hoge() {}

    struct Task
    {
        std::atomic<bool> valid = false;
        std::function<void(void)> func;

        Task() 
        {
        }
        Task(const Task& t)
        {
            valid.store(t.valid.load());
            func = t.func;
        }
    };
    std::vector<Task> thraed_task;

    std::atomic<uint32_t> prev_index;
}


//
void preprocess(
    const float* vertices,
    size_t numVerts,
    const float* normals,
    size_t numNormals,
    const uint32_t* indices,
    size_t numFace)
{
    impl::init();

#ifdef MYTHREADING
    // スレッド
    impl::thraed_pool.resize(impl::NUM_THREAD);
    impl::thraed_task.resize(impl::NUM_THREAD);
    auto& thraed_task = impl::thraed_task;
    for (int i = 0; i < impl::NUM_THREAD; ++i)
    {
        impl::thraed_pool[i] = std::thread([i, &thraed_task]()
        {
            while (1)
            {
                // std::this_thread::sleep_for(std::chrono::milliseconds(1));
                if (thraed_task[i].valid.load())
                {
                    thraed_task[i].func();
                    thraed_task[i].valid.store(false);
                }
            }
        });
    }
#endif


    impl::vertices = std::vector<float>(vertices, vertices + numVerts * 3);
    impl::normals = std::vector<float>(normals, normals + numNormals * 3);
    impl::triangles.resize(numFace);
    impl::simd_triangles.resize((numFace / PackedTriangle) + 1);

    alignas(64) float x[PackedTriangle * 3] = {};
    alignas(64) float y[PackedTriangle * 3] = {};
    alignas(64) float z[PackedTriangle * 3] = {};

    int simd_index = 0;

    for (size_t i = 0; i < numFace; ++i)
    {
        const int offset = i % PackedTriangle;

        for (int a = 0; a < 3; ++a)
        {
            impl::triangles[i].v[0][a] = impl::vertices[indices[i * 6 + 0] * 3 + a];
            impl::triangles[i].v[1][a] = impl::vertices[indices[i * 6 + 2] * 3 + a];
            impl::triangles[i].v[2][a] = impl::vertices[indices[i * 6 + 4] * 3 + a];
            impl::triangles[i].n[0][a] = impl::normals[indices[i * 6 + 1] * 3 + a];
            impl::triangles[i].n[1][a] = impl::normals[indices[i * 6 + 3] * 3 + a];
            impl::triangles[i].n[2][a] = impl::normals[indices[i * 6 + 5] * 3 + a];

        }

        for (int v = 0; v < 3; ++v)
        {
            x[offset + PackedTriangle * v] = impl::triangles[i].v[v][0];
            y[offset + PackedTriangle * v] = impl::triangles[i].v[v][1];
            z[offset + PackedTriangle * v] = impl::triangles[i].v[v][2];
        }

        if (offset == PackedTriangle - 1 || i == numFace - 1)
        {
            auto& t = impl::simd_triangles[simd_index++];

            for (int v = 0; v < 3; ++v)
            {
                t.x[v] = _mm512_load_ps(x + PackedTriangle * v);
                t.y[v] = _mm512_load_ps(y + PackedTriangle * v);
                t.z[v] = _mm512_load_ps(z + PackedTriangle * v);
            }
            t.triangle_offset = i - (PackedTriangle - 1);

            std::fill(x, x + PackedTriangle * 3, 0);
            std::fill(y, y + PackedTriangle * 3, 0);
            std::fill(z, z + PackedTriangle * 3, 0);
        }
    }
}


namespace rt
{
    struct Result
    {
        bool hit = false;
        float t = 0;
        float u = 0;
        float v = 0;
    };

    Result ray_vs_triangle(const math::Float3& org, const math::Float3& dir, const impl::Triangle* triangle)
    {
        constexpr float EPSILON = 0.0000001;

        auto v0 = triangle->v[0];
        auto v1 = triangle->v[1];
        auto v2 = triangle->v[2];

        math::Float3 e1, e2, h, s, q;
        float a, f, u, v;
        e1 = v1 - v0;
        e2 = v2 - v0;
        h = math::cross(dir, e2);
        a = math::dot(e1, h);
        if (a > -EPSILON && a < EPSILON)
            return {};    // This ray is parallel to this triangle.
        f = 1.0 / a;
        s = org - v0;
        u = f * math::dot(s, h);
        if (u < 0.0 || u > 1.0)
            return {};
        q = math::cross(s, e1);
        v = f * math::dot(dir, q);
        if (v < 0.0 || u + v > 1.0)
            return {};
        // At this stage we can compute t to find out where the intersection point is on the line.
        float t = f * math::dot(e2, q);
        if (t > EPSILON) // ray intersection
        {
            return { true, t, u, v };
        }
        else // This means that there is a line intersection but not a ray intersection.
            return {};
    }
}

void intersect(
    Ray* rays,
    size_t numRay,
    bool hitany)
{
    for (int rayi = 0; rayi < numRay; ++rayi)
    {
        auto* current_ray = &rays[rayi];

        const math::Float3 org(current_ray->pos);
        const math::Float3 dir(current_ray->dir);
        current_ray->isisect = false;

        uint32_t tid = 0;
        float nearest_t = std::numeric_limits<float>::infinity();
        float u, v;
#ifdef MYTHREADING
        // コア数で並列化
        impl::Hitpoint thread_hitpoint[impl::NUM_THREAD];
        impl::Hitpoint final_hitpoint;

        struct TaskDiv
        {
            int begin_index;
            int end_index;
        };
        TaskDiv tasks[impl::NUM_THREAD];
        const auto Block = impl::simd_triangles.size() / impl::NUM_THREAD;
        for (int tid = 0; tid < impl::NUM_THREAD; ++tid)
        {
            tasks[tid].begin_index = tid * Block;
            tasks[tid].end_index = (tid + 1) * Block;
        }
        tasks[impl::NUM_THREAD - 1].end_index = impl::simd_triangles.size();

        // 実行
        for (int tid = 0; tid < impl::NUM_THREAD; ++tid)
        {
            auto& current_task = tasks[tid];
            auto& hitpoint = thread_hitpoint[tid];
#if 1
            impl::thraed_task[tid].func = [&hitpoint, &current_ray, hitany, current_task]() {
                alignas(64) float org_x[PackedTriangle];
                alignas(64) float org_y[PackedTriangle];
                alignas(64) float org_z[PackedTriangle];
                alignas(64) float dir_x[PackedTriangle];
                alignas(64) float dir_y[PackedTriangle];
                alignas(64) float dir_z[PackedTriangle];
                
                for (int i = 0; i < PackedTriangle; ++i)
                {
                    org_x[i] = current_ray->pos[0];
                    org_y[i] = current_ray->pos[1];
                    org_z[i] = current_ray->pos[2];
                
                    dir_x[i] = current_ray->dir[0];
                    dir_y[i] = current_ray->dir[1];
                    dir_z[i] = current_ray->dir[2];
                }
                
                __m512 avx_org[3];
                avx_org[0] = _mm512_load_ps(org_x);
                avx_org[1] = _mm512_load_ps(org_y);
                avx_org[2] = _mm512_load_ps(org_z);
                __m512 avx_dir[3];
                avx_dir[0] = _mm512_load_ps(dir_x);
                avx_dir[1] = _mm512_load_ps(dir_y);
                avx_dir[2] = _mm512_load_ps(dir_z);
                
                // impl::Hitpoint hitpoint;
                
                for (size_t i = current_task.begin_index; i < current_task.end_index; ++i)
                {
                    impl::ray_vs_triangle(current_ray->tnear, current_ray->tfar, avx_org, avx_dir,
                        impl::simd_triangles[i], &hitpoint);
                
                    if (hitpoint.hit)
                    {
                        if (hitany)
                        {
                            break;
                        }
                    }
                }
            };
#endif
            impl::thraed_task[tid].valid.store(true);
        }

        // thread待ち
        for (int tid = 0; tid < impl::NUM_THREAD; ++tid)
        {
            while (impl::thraed_task[tid].valid.load()) {
                // std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }

        // reduction
        for (int tid = 0; tid < impl::NUM_THREAD; ++tid)
        {
            if (thread_hitpoint[tid].hit && 
                thread_hitpoint[tid].distance > final_hitpoint.distance)
            {
                final_hitpoint = thread_hitpoint[tid];
            }
        }

        if (final_hitpoint.hit)
        {
            current_ray->isisect = true;
            nearest_t = final_hitpoint.distance;
            u = final_hitpoint.b1;
            v = final_hitpoint.b2;
            tid = final_hitpoint.index;
        }
#else
        // SIMD
        {
            alignas(64) float org_x[PackedTriangle];
            alignas(64) float org_y[PackedTriangle];
            alignas(64) float org_z[PackedTriangle];
            alignas(64) float dir_x[PackedTriangle];
            alignas(64) float dir_y[PackedTriangle];
            alignas(64) float dir_z[PackedTriangle];

            for (int i = 0; i < PackedTriangle; ++i)
            {
                org_x[i] = current_ray->pos[0];
                org_y[i] = current_ray->pos[1];
                org_z[i] = current_ray->pos[2];

                dir_x[i] = current_ray->dir[0];
                dir_y[i] = current_ray->dir[1];
                dir_z[i] = current_ray->dir[2];
            }

            __m512 avx_org[3];
            avx_org[0] = _mm512_load_ps(org_x);
            avx_org[1] = _mm512_load_ps(org_y);
            avx_org[2] = _mm512_load_ps(org_z);
            __m512 avx_dir[3];
            avx_dir[0] = _mm512_load_ps(dir_x);
            avx_dir[1] = _mm512_load_ps(dir_y);
            avx_dir[2] = _mm512_load_ps(dir_z);

            impl::Hitpoint hitpoint;

            struct Span
            {
                size_t begin;
                size_t end;
            };

            Span span[2];

            span[0].begin = 0;
            span[0].end = impl::simd_triangles.size() / 2;

            span[1].begin = span[0].end;
            span[1].end = impl::simd_triangles.size();

            alignas(64) float near_t_data[PackedTriangle];
            alignas(64) float far_t_data[PackedTriangle];
            std::fill(near_t_data, near_t_data + PackedTriangle, current_ray->tnear);
            std::fill(far_t_data, far_t_data + PackedTriangle, current_ray->tfar);
            __m512 near_t = _mm512_load_ps(near_t_data);
            __m512 far_t = _mm512_load_ps(far_t_data);

            alignas(64) float current_distance_data[PackedTriangle];
            std::fill(current_distance_data, current_distance_data + PackedTriangle, hitpoint.distance);
            __m512 current_distance = _mm512_load_ps(current_distance_data);

            for (int s = 0; s < 2; ++s)
            {
                for (size_t i = span[s].begin; i < span[s].end; ++i)
                {
                    impl::ray_vs_triangle(current_distance, near_t, far_t, avx_org, avx_dir, impl::simd_triangles[i], &hitpoint);
//                    impl::ray_vs_triangle(current_distance, current_ray->tnear, current_ray->tfar, avx_org, avx_dir, impl::simd_triangles[i], &hitpoint);

                    if (hitpoint.hit)
                    {
                        if (hitany)
                        {
                            goto END;
                        }
                    }
                }
            }
            END:
            
#if 0
            for (size_t i = 0; i < impl::simd_triangles.size(); ++i)
            {
                impl::ray_vs_triangle(current_ray->tnear, current_ray->tfar, avx_org, avx_dir,
                    impl::simd_triangles[i], &hitpoint);

                if (hitpoint.hit)
                {
                    if (hitany)
                    {
                        break;
                    }
                }
            }
#endif

            if (hitpoint.hit)
            {
                current_ray->isisect = true;
                nearest_t = hitpoint.distance;
                u = hitpoint.b1;
                v = hitpoint.b2;
                tid = hitpoint.index;
            }
        }
#endif

#if 0
        for (size_t i = 0; i < impl::triangles.size(); ++i)
        {
            auto r = rt::ray_vs_triangle(org, dir, &impl::triangles[i]);
            if (r.hit)
            {
                if (current_ray->tnear <= r.t && r.t <= current_ray->tfar)
                {
                    if (r.t < nearest_t)
                    {
                        current_ray->isisect = true;
                        nearest_t = r.t;
                        u = r.u;
                        v = r.v;
                        tid = i;

                        if (hitany)
                        {
                            break;
                        }
                    }
                }
            }
        }
#endif

        if (current_ray->isisect)
        {
            current_ray->faceid = tid;
            auto& triangle = impl::triangles[tid];

            auto pos =
                (1 - u - v) * triangle.v[0] +
                u * triangle.v[1] +
                v * triangle.v[2];
            std::copy(&pos[0], &pos[0] + 3, current_ray->isect);

            auto normal =
                (1 - u - v) * triangle.n[0] +
                u * triangle.n[1] +
                v * triangle.n[2];

            std::copy(&normal[0], &normal[0] + 3, rays->ns);

        }
    }

}

#else

//
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
//
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>
#include <array>
#include <iostream>
#include <type_traits>
//
#include "rayrun.hpp"
#include "vec3.h"

#include "hlib/rt/objMesh.h"

//
BOOL APIENTRY DllMain(HMODULE hModule,
    DWORD  ul_reason_for_call,
    LPVOID lpReserved
)
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}

namespace impl
{
    std::vector<float> vertices;
    std::vector<float> normals;
    struct Triangle
    {
        hstd::Float3 v[3];
        hstd::Float3 n[3];
    };
    std::vector<Triangle> triangles;
    std::vector<hstd::rt::RefTriangle> ref_triangles;

    hstd::rt::QBVH qbvh;
}

bool neverUseOpenMP()
{
    return false;
}

//
void preprocess(
    const float* vertices,
    size_t numVerts,
    const float* normals,
    size_t numNormals,
    const uint32_t* indices,
    size_t numFace)
{
    impl::vertices = std::vector<float>(vertices, vertices + numVerts * 3);
    impl::normals = std::vector<float>(normals, normals + numNormals * 3);
    impl::triangles.resize(numFace);
    impl::ref_triangles.resize(numFace);
    for (size_t i = 0; i < numFace; ++i)
    {
        for (int a = 0; a < 3; ++a)
        {
            impl::triangles[i].v[0][a] = impl::vertices[indices[i * 6 + 0] * 3 + a];
            impl::triangles[i].v[1][a] = impl::vertices[indices[i * 6 + 2] * 3 + a];
            impl::triangles[i].v[2][a] = impl::vertices[indices[i * 6 + 4] * 3 + a];
            impl::triangles[i].n[0][a] = impl::normals[indices[i * 6 + 1] * 3 + a];
            impl::triangles[i].n[1][a] = impl::normals[indices[i * 6 + 3] * 3 + a];
            impl::triangles[i].n[2][a] = impl::normals[indices[i * 6 + 5] * 3 + a];
        }

        for (int a = 0; a < 3; ++a)
        {
            impl::ref_triangles[i].p[a] = &impl::triangles[i].v[a];
        }
        impl::ref_triangles[i].original_triangle_index = i;
    }

    impl::qbvh.build(impl::ref_triangles);
}

void intersect(
    Ray* rays,
    size_t numRay,
    bool hitany)
{
    for (int iray = 0; iray < numRay; ++iray)
    {
        auto* current_ray = &rays[iray];
        hstd::rt::Ray ray;
        std::copy(current_ray->pos, current_ray->pos + 3, &ray.org[0]);
        std::copy(current_ray->dir, current_ray->dir + 3, &ray.dir[0]);

        hstd::rt::Hitpoint hp;
        current_ray->isisect = impl::qbvh.intersect(ray, &hp, hitany, current_ray->tnear, current_ray->tfar);

        if (current_ray->isisect)
        {
            const float u = hp.b1;
            const float v = hp.b2;
            current_ray->faceid = hp.triangle_index;
            auto& triangle = impl::triangles[current_ray->faceid];

            auto pos =
                (1 - u - v) * triangle.v[0] +
                u * triangle.v[1] +
                v * triangle.v[2];
            std::copy(&pos[0], &pos[0] + 3, current_ray->isect);

                
#if 0
            auto normal = hstd::cross(triangle.v[1] - triangle.v[0], triangle.v[2] - triangle.v[0]);
            normal = hstd::normalize(normal);
#else
            auto normal =
                (1 - u - v) * triangle.n[0] +
                u * triangle.n[1] +
                v * triangle.n[2];
#endif
            std::copy(&normal[0], &normal[0] + 3, current_ray->ns);
        }
    }
}

#endif