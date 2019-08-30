
#if 0
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
        math::Float3 v[3];
        math::Float3 n[3];
    };
    std::vector<Triangle> triangles;
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
    for (size_t i = 0; i < numFace; ++i)
    {
        for (int a = 0; a < 3; ++a)
        {
            impl::triangles[i].v[0][a] = impl::vertices[indices[i * 6 + 0] * 3 + a];
            impl::triangles[i].v[1][a] = impl::vertices[indices[i * 6 + 2] * 3 + a];
            impl::triangles[i].v[2][a] = impl::vertices[indices[i * 6 + 4] * 3 + a];
#if 0
            impl::triangles[i].n[0][a] = impl::normals[indices[i * 6 + 1] * 3 + a];
            impl::triangles[i].n[1][a] = impl::normals[indices[i * 6 + 3] * 3 + a];
            impl::triangles[i].n[2][a] = impl::normals[indices[i * 6 + 5] * 3 + a];
#endif
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
    const math::Float3 org(rays->pos);
    const math::Float3 dir(rays->dir);
    rays->isisect = false;

    uint32_t tid = 0;
    float nearest_t = std::numeric_limits<float>::infinity();
    float u, v;
    for (size_t i = 0; i < impl::triangles.size(); ++i)
    {
        auto r = rt::ray_vs_triangle(org, dir, &impl::triangles[i]);
        if (r.hit)
        {
            if (rays->tnear <= r.t && r.t <= rays->tfar)
            {
                if (r.t < nearest_t)
                {
                    rays->isisect = true;
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

    if (rays->isisect)
    {
        auto pos = org + nearest_t * dir;
        std::copy(&pos[0], &pos[0] + 3, rays->isect);
        rays->faceid = tid;

        auto& triangle = impl::triangles[tid];

        auto normal = math::cross(triangle.v[1] - triangle.v[0], triangle.v[2] - triangle.v[0]);
        normal = math::normalize(normal);
#if 0
        auto normal = 
            v * triangle.n[1] +
            (1 - u) * (1 - v) * triangle.n[0] +
            u * (1 - v) * triangle.v[2];
#endif
        std::copy(&normal[0], &normal[0] + 3, rays->ns);
    }
}

#endif



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
#if 0
            impl::triangles[i].n[0][a] = impl::normals[indices[i * 6 + 1] * 3 + a];
            impl::triangles[i].n[1][a] = impl::normals[indices[i * 6 + 3] * 3 + a];
            impl::triangles[i].n[2][a] = impl::normals[indices[i * 6 + 5] * 3 + a];
#endif
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
            auto pos = ray.org + hp.distance * ray.dir;
            std::copy(&pos[0], &pos[0] + 3, current_ray->isect);
            current_ray->faceid = hp.triangle_index;

            auto& triangle = impl::triangles[current_ray->faceid];

            auto normal = hstd::cross(triangle.v[1] - triangle.v[0], triangle.v[2] - triangle.v[0]);
            normal = hstd::normalize(normal);
#if 0
            auto normal =
                v * triangle.n[1] +
                (1 - u) * (1 - v) * triangle.n[0] +
                u * (1 - v) * triangle.v[2];
#endif
            std::copy(&normal[0], &normal[0] + 3, current_ray->ns);
        }
    }
}

#endif