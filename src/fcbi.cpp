#include <cstdint>
#include <algorithm>
#include <emmintrin.h>
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <avisynth.h>


typedef IScriptEnvironment ise_t;


static inline int mean(uint8_t x, uint8_t y) noexcept
{
    return (x + y + 1) / 2;
}

#if 0
static void phase1_c(
    const uint8_t* srcp, uint8_t* dstp, const int width, const int height,
    const int spitch, const int dpitch) noexcept
{
    dstp[-2] = dstp[-1] = srcp[0];
    for (int x = 0; x < width - 1; ++x) {
        dstp[2 * x] = srcp[x];
        dstp[2 * x + 1] = mean(srcp[x], srcp[x + 1]);
    }
    dstp[2 * width - 2] = dstp[2 * width - 1] = dstp[2 * width] = srcp[width - 1];
    srcp += spitch;
    dstp += 2 * dpitch;

    for (int y = 1; y < height - 1; ++y) {
        dstp[-2] = dstp[-1] = srcp[0];

        uint16_t* d16 = reinterpret_cast<uint16_t*>(dstp);
        for (int x = 0; x < width - 1; ++x) {
            d16[x] = static_cast<uint16_t>(srcp[x]);
        }
        dstp[2 * width - 2] = dstp[2 * width - 1] = dstp[2 * width] = srcp[width - 1];
        srcp += spitch;
        dstp += 2 * dpitch;
    }

    dstp[-2] = dstp[-1] = srcp[0];
    for (int x = 0; x < width - 1; ++x) {
        dstp[2 * x] = srcp[x];
        dstp[2 * x + 1] = mean(srcp[x], srcp[x + 1]);
    }
    dstp[2 * width - 2] = dstp[2 * width - 1] = dstp[2 * width] = srcp[width - 1];
    dstp -= 2;
    memcpy(dstp + dpitch, dstp, 2 * width + 4);
    dstp += dpitch;
    memcpy(dstp + dpitch, dstp, 2 * width + 4);
}
#endif

static void phase1_sse2(
    const uint8_t* srcp, uint8_t* dstp, const int width, const int height,
    const int spitch, const int dpitch) noexcept
{
    const __m128i zero = _mm_setzero_si128();

    dstp[-2] = dstp[-1] = srcp[0];
    for (int x = 0; x < width - 1; x += 16) {
        __m128i s0 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp + x));
        __m128i s1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + x + 1));
        s1 = _mm_avg_epu8(s0, s1);
        __m128i d0 = _mm_unpacklo_epi8(s0, s1);
        __m128i d1 = _mm_unpackhi_epi8(s0, s1);
        _mm_store_si128(reinterpret_cast<__m128i*>(dstp + 2 * x), d0);
        _mm_store_si128(reinterpret_cast<__m128i*>(dstp + 2 * x + 16), d1);
    }
    dstp[2 * width - 2] = dstp[2 * width - 1] = dstp[2 * width] = srcp[width - 1];
    srcp += spitch;
    dstp += 2 * dpitch;

    for (int y = 1; y < height - 1; ++y) {
        dstp[-2] = dstp[-1] = srcp[0];
        for (int x = 0; x < width - 1; x += 16) {
            __m128i s0 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp + x));
            __m128i d0 = _mm_unpacklo_epi8(s0, zero);
            __m128i d1 = _mm_unpackhi_epi8(s0, zero);
            _mm_store_si128(reinterpret_cast<__m128i*>(dstp + 2 * x), d0);
            _mm_store_si128(reinterpret_cast<__m128i*>(dstp + 2 * x + 16), d1);
        }
        dstp[2 * width - 2] = dstp[2 * width - 1] = dstp[2 * width] = srcp[width - 1];
        srcp += spitch;
        dstp += 2 * dpitch;
    }

    dstp[-2] = dstp[-1] = srcp[0];
    for (int x = 0; x < width - 1; x += 16) {
        __m128i s0 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp + x));
        __m128i s1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + x + 1));
        s1 = _mm_avg_epu8(s0, s1);
        __m128i d0 = _mm_unpacklo_epi8(s0, s1);
        __m128i d1 = _mm_unpackhi_epi8(s0, s1);
        _mm_store_si128(reinterpret_cast<__m128i*>(dstp + 2 * x), d0);
        _mm_store_si128(reinterpret_cast<__m128i*>(dstp + 2 * x + 16), d1);
    }
    dstp[2 * width - 2] = dstp[2 * width - 1] = dstp[2 * width] = srcp[width - 1];
    dstp -= 2;
    memcpy(dstp + dpitch, dstp, 2 * width + 4);
    dstp += dpitch;
    memcpy(dstp + dpitch, dstp, 2 * width + 4);
}


static inline int abs_diff(int x, int y)
{
    return x > y ? x - y : y - x;
}


static inline bool is_edge(int v1, int v2, int p1, int p2, int tm)
{
    if (abs_diff(v1, v2) < tm) {
        return false;
    }
    return !(v1 < tm && v2 < tm && abs_diff(p1, p2) < tm * 2);
}


template <bool EDGE>
static void phase2_c(
    uint8_t* ptr, const int width, const int height, const int pitch,
    const int tm) noexcept
{
    const int p2 = 2 * pitch;
    const uint8_t* s0 = ptr;
    const uint8_t* s1 = s0;
    const uint8_t* s2 = s1 + p2;
    const uint8_t* s3 = s2 + p2;

    uint8_t* dstp = ptr + pitch;

    for (int y = 1; y < height - 1; y += 2) {
        dstp[0] = mean(s1[0], s2[0]);
        for (int x = 1; x < width - 2; x += 2) {
            int p1 = s1[x - 1] + s2[x + 1];
            int p2 = s1[x + 1] + s2[x - 1];
            if (EDGE) {
                int v1 = abs_diff(s1[x - 1], s2[x + 1]);
                int v2 = abs_diff(s1[x + 1], s2[x - 1]);
                if (is_edge(v1, v2, p1, p2, tm)) {
                    if (v1 < v2) {
                        dstp[x] = (p1 + 1) / 2;
                    } else {
                        dstp[x] = (p2 + 1) / 2;
                    }
                    continue;
                }
            }
            int h1 = s0[x + 1] + s1[x + 3] + s2[x - 3] + s3[x - 1] + p1 - 3 * p2;
            int h2 = s0[x - 1] + s1[x - 3] + s2[x + 3] + s3[x + 1] + p2 - 3 * p1;
            if (std::abs(h1) < std::abs(h2)) {
                dstp[x] = (p1 + 1) / 2;
            } else {
                dstp[x] = (p2 + 1) / 2;
            }
        }
        dstp[width - 2] = dstp[width - 1] = mean(s1[width - 2], s2[width - 2]);

        s0 = s1;
        s1 = s2;
        s2 = s3;
        s3 += p2;
        dstp += p2;
    }
}


template <bool EDGE>
static void phase3_c(
    uint8_t* ptr, const int width, const int height, const int pitch,
    const int tm) noexcept
{
    const uint8_t* s0 = ptr;
    const uint8_t* s1 = ptr;
    const uint8_t* s2 = s1 + pitch;
    const uint8_t* s3 = s2 + pitch;
    const uint8_t* s4 = s3 + pitch;

    uint8_t* dstp = ptr + pitch;

    for (int y = 1; y < height - 2; ++y) {
        for (int x = 1 + (y & 1); x < width - 2; x += 2) {
            int p1 = s2[x - 1] + s2[x + 1];
            int p2 = s1[x] + s3[x];
            if (EDGE) {
                int v1 = abs_diff(s2[x - 1], s2[x + 1]);
                int v2 = abs_diff(s1[x], s3[x]);
                if (is_edge(v1, v2, p1, p2, tm)) {
                    if (v1 < v2) {
                        dstp[x] = (p1 + 1) / 2;
                    } else {
                        dstp[x] = (p2 + 1) / 2;
                    }
                    continue;
                }
            }
            int h1 = s0[x - 1] + s0[x + 1] + s4[x - 1] + s4[x + 1] + p1 - 3 * p2;
            int h2 = s1[x - 2] + s1[x + 2] + s3[x - 2] + s3[x + 2] + p2 - 3 * p1;
            if (std::abs(h1) < std::abs(h2)) {
                dstp[x] = (p1 + 1) / 2;
            } else {
                dstp[x] = (p2 + 1) / 2;
            }
        }
        s0 = s1;
        s1 = s2;
        s2 = s3;
        s3 = s4;
        s4 += pitch;
        dstp += pitch;
    }
}


class FCBI : public GenericVideoFilter {
    const bool edge;
    const int tm;
    VideoInfo vit;
    int numPlanes;
public:
    FCBI(PClip child, bool edge, int tm);
    ~FCBI() {}
    PVideoFrame __stdcall GetFrame(int n, ise_t* env);
    int __stdcall SetCacheHints(int hints, int)
    {
        return hints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
    }
    static AVSValue __cdecl create(AVSValue args, void*, ise_t* env);
};


FCBI::FCBI(PClip _c, bool _e, int _t) : GenericVideoFilter(_c), edge(_e), tm(_t)
{
    numPlanes = vi.IsY8() ? 1 : 3;

    vi.width *= 2;
    vi.height *= 2;

    vit = vi;
    vit.pixel_type = VideoInfo::CS_Y8;
    vit.width = (vit.width + 4 + 31) & ~31;
    vit.height += 2;
}


PVideoFrame __stdcall FCBI::GetFrame(int n, ise_t* env)
{
    const int planes[] = { PLANAR_Y, PLANAR_U, PLANAR_V };

    auto src = child->GetFrame(n, env);
    auto tmp = env->NewVideoFrame(vit);
    auto dst = env->NewVideoFrame(vi);

    const int tpitch = tmp->GetPitch();
    uint8_t* tmpp = tmp->GetWritePtr() + tpitch;

    for (int p = 0; p < numPlanes; ++p) {
        const int plane = planes[p];
        const int width = dst->GetRowSize(plane);
        const int height = dst->GetHeight(plane);

        phase1_sse2(src->GetReadPtr(plane), tmpp, src->GetRowSize(plane),
                    src->GetHeight(plane), src->GetPitch(plane), tpitch);
        if (edge) {
            phase2_c<true>(tmpp, width, height, tpitch, tm);
            phase3_c<true>(tmpp, width, height, tpitch, tm);
        } else {
            phase2_c<false>(tmpp, width, height, tpitch, 0);
            phase3_c<false>(tmpp, width, height, tpitch, 0);
        }
        env->BitBlt(dst->GetWritePtr(plane), dst->GetPitch(plane), tmpp,
                    tpitch, width, height);
    }

    return dst;
}


AVSValue __cdecl FCBI::create(AVSValue args, void*, ise_t* env)
{
    auto validate = [env](bool cond, const char* msg) {
        if (cond) {
            env->ThrowError("FCBI: %s.", msg);
        }
    };

    PClip c = args[0].AsClip();
    const VideoInfo& vi = c->GetVideoInfo();

    validate(!vi.IsPlanar() || vi.IsRGB(), "Input clip is not planar YUV format");
    validate(vi.BytesFromPixels(1) != 1, "Bit depth of input clip is not 8 bit");
    validate(vi.pixel_type == VideoInfo::CS_YUVA, "Alpha is unsupported");
    validate(vi.IsYV411(), "Input clip is unsupported format");
    validate(vi.width < 16 || vi.height < 16, "Input clip is too small");

    int tm = std::min(std::max(args[2].AsInt(30), 0), 255);

    return new FCBI(c, args[1].AsBool(false), tm);
}


const AVS_Linkage* AVS_linkage = nullptr;


extern "C" __declspec(dllexport) const char* __stdcall
AvisynthPluginInit3(ise_t* env, const AVS_Linkage* const vectors)
{
    AVS_linkage = vectors;
    env->AddFunction("FCBI", "c[ed]b[tm]i", FCBI::create, nullptr);
    return "FCBI for avisynth ver 0.0.0";
}
