/*
This file is part of Mitsuba, a physically based rendering system.

Copyright (c) 2007-2014 by Wenzel Jakob and others.

Mitsuba is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License Version 3
as published by the Free Software Foundation.

Mitsuba is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
*/
// clang-format off
#include <mitsuba/core/fresolver.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/hw/basicshader.h>
#include "microfacet.h"
#include "ior.h"
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <numeric>
// clang-format on

MTS_NAMESPACE_BEGIN

/*!\plugin{roughconductor}{Rough conductor material}
 * \order{7}
 * \icon{bsdf_roughconductor}
 * \parameters{
 *     \parameter{distribution}{\String}{
 *          Specifies the type of microfacet normal distribution
 *          used to model the surface roughness.
 *          \vspace{-1mm}
 *       \begin{enumerate}[(i)]
 *           \item \code{beckmann}: Physically-based distribution derived from
 *               Gaussian random surfaces. This is the default.\vspace{-1.5mm}
 *           \item \code{ggx}: The GGX \cite{Walter07Microfacet} distribution
 * (also known as Trowbridge-Reitz \cite{Trowbridge19975Average} distribution)
 *               was designed to better approximate the long tails observed in
 * measurements of ground surfaces, which are not modeled by the Beckmann
 * distribution. \vspace{-1.5mm} \item \code{phong}: Anisotropic Phong
 * distribution by Ashikhmin and Shirley \cite{Ashikhmin2005Anisotropic}. In
 * most cases, the \code{ggx} and \code{beckmann} distributions should be
 * preferred, since they provide better importance sampling and accurate
 * shadowing/masking computations. \vspace{-4mm} \end{enumerate}
 *     }
 *     \parameter{alpha, alphaU, alphaV}{\Float\Or\Texture}{
 *         Specifies the roughness of the unresolved surface micro-geometry
 *         along the tangent and bitangent directions. When the Beckmann
 *         distribution is used, this parameter is equal to the
 *         \emph{root mean square} (RMS) slope of the microfacets.
 *         \code{alpha} is a convenience parameter to initialize both
 *         \code{alphaU} and \code{alphaV} to the same value. \default{0.1}.
 *     }
 *     \parameter{material}{\String}{Name of a material preset, see
 *           \tblref{conductor-iors}.\!\default{\texttt{Cu} / copper}}
 *     \parameter{eta, k}{\Spectrum}{Real and imaginary components of the
 * material's index of refraction \default{based on the value of
 * \texttt{material}}} \parameter{extEta}{\Float\Or\String}{ Real-valued index
 * of refraction of the surrounding dielectric, or a material name of a
 * dielectric \default{\code{air}}
 *     }
 *     \parameter{sampleVisible}{\Boolean}{
 *         Enables a sampling technique proposed by Heitz and
 * D'Eon~\cite{Heitz1014Importance}, which focuses computation on the visible
 * parts of the microfacet normal distribution, considerably reducing variance
 * in some cases. \default{\code{true}, i.e. use visible normal sampling}
 *     }
 *     \parameter{specular\showbreak Reflectance}{\Spectrum\Or\Texture}{Optional
 *         factor that can be used to modulate the specular reflection
 * component. Note that for physical realism, this parameter should never be
 * touched. \default{1.0}}
 * }
 * \vspace{3mm}
 * This plugin implements a realistic microfacet scattering model for rendering
 * rough conducting materials, such as metals. It can be interpreted as a fancy
 * version of the Cook-Torrance model and should be preferred over
 * heuristic models like \pluginref{phong} and \pluginref{ward} if possible.
 * \renderings{
 *     \rendering{Rough copper (Beckmann, $\alpha=0.1$)}
 *     	   {bsdf_roughconductor_copper.jpg}
 *     \rendering{Vertically brushed aluminium (Anisotropic Phong,
 *         $\alpha_u=0.05,\ \alpha_v=0.3$), see
 *         \lstref{roughconductor-aluminium}}
 *         {bsdf_roughconductor_anisotropic_aluminium.jpg}
 * }
 *
 * Microfacet theory describes rough surfaces as an arrangement of unresolved
 * and ideally specular facets, whose normal directions are given by a
 * specially chosen \emph{microfacet distribution}. By accounting for shadowing
 * and masking effects between these facets, it is possible to reproduce the
 * important off-specular reflections peaks observed in real-world measurements
 * of such materials.
 *
 * This plugin is essentially the ``roughened'' equivalent of the (smooth)
 * plugin \pluginref{conductor}. For very low values of $\alpha$, the two will
 * be identical, though scenes using this plugin will take longer to render
 * due to the additional computational burden of tracking surface roughness.
 *
 * The implementation is based on the paper ``Microfacet Models
 * for Refraction through Rough Surfaces'' by Walter et al.
 * \cite{Walter07Microfacet}. It supports three different types of microfacet
 * distributions and has a texturable roughness parameter.
 * To facilitate the tedious task of specifying spectrally-varying index of
 * refraction information, this plugin can access a set of measured materials
 * for which visible-spectrum information was publicly available
 * (see \tblref{conductor-iors} for the full list).
 * There is also a special material profile named \code{none}, which disables
 * the computation of Fresnel reflectances and produces an idealized
 * 100% reflecting mirror.
 *
 * When no parameters are given, the plugin activates the default settings,
 * which describe copper with a medium amount of roughness modeled using a
 * Beckmann distribution.
 *
 * To get an intuition about the effect of the surface roughness parameter
 * $\alpha$, consider the following approximate classification: a value of
 * $\alpha=0.001-0.01$ corresponds to a material with slight imperfections
 * on an otherwise smooth surface finish, $\alpha=0.1$ is relatively rough,
 * and $\alpha=0.3-0.7$ is \emph{extremely} rough (e.g. an etched or ground
 * finish). Values significantly above that are probably not too realistic.
 * \vspace{4mm}
 * \begin{xml}[caption={A material definition for brushed aluminium},
 * label=lst:roughconductor-aluminium] <bsdf type="roughconductor"> <string
 * name="material" value="Al"/> <string name="distribution" value="phong"/>
 *     <float name="alphaU" value="0.05"/>
 *     <float name="alphaV" value="0.3"/>
 * </bsdf>
 * \end{xml}
 *
 * \subsubsection*{Technical details}
 * All microfacet distributions allow the specification of two distinct
 * roughness values along the tangent and bitangent directions. This can be
 * used to provide a material with a ``brushed'' appearance. The alignment
 * of the anisotropy will follow the UV parameterization of the underlying
 * mesh. This means that such an anisotropic material cannot be applied to
 * triangle meshes that are missing texture coordinates.
 *
 * \label{sec:visiblenormal-sampling}
 * Since Mitsuba 0.5.1, this plugin uses a new importance sampling technique
 * contributed by Eric Heitz and Eugene D'Eon, which restricts the sampling
 * domain to the set of visible (unmasked) microfacet normals. The previous
 * approach of sampling all normals is still available and can be enabled
 * by setting \code{sampleVisible} to \code{false}.
 * Note that this new method is only available for the \code{beckmann} and
 * \code{ggx} microfacet distributions. When the \code{phong} distribution
 * is selected, the parameter has no effect.
 *
 * When rendering with the Phong microfacet distribution, a conversion is
 * used to turn the specified Beckmann-equivalent $\alpha$ roughness value
 * into the exponent parameter of this distribution. This is done in a way,
 * such that the same value $\alpha$ will produce a similar appearance across
 * different microfacet distributions.
 *
 * When using this plugin, you should ideally compile Mitsuba with support for
 * spectral rendering to get the most accurate results. While it also works
 * in RGB mode, the computations will be more approximate in nature.
 * Also note that this material is one-sided---that is, observed from the
 * back side, it will be completely black. If this is undesirable,
 * consider using the \pluginref{twosided} BRDF adapter.
 */
#define THRESHOLD 1e-4

static inline float generateRandomNumber()
{
	const float U = ((float)rand()) / (float)RAND_MAX;
	return U;
}

class SegmentTerm {
public:
    float              lambdao;
    int                N = 0;
    float              m = 1;
    std::vector<float> e{}, g{}, l{};
    explicit SegmentTerm(float lambda) : lambdao(lambda) {}
    void addBounce(float lambda) {
        if (lambda < 0) { // Downwards
            l.push_back(-lambda);
            e.push_back(1.0f / (lambdao + l.back()));
            g.push_back(0);
            m *= e.back(); // cache the results when sampled rays are only downwards
            N++;
        } else { // Upwards
            if (m == 0) {
                g.back() /= (lambda + l.back());
            } else { // Initialize g
                g.back() = 1.0f / (lambda + l.back());
                m        = 0;
            }
            for (int i = N - 2; i >= 0; i--) {
                g[i] = (g[i] + g[i + 1]) / (lambda + l[i]);
            }
        }
    }
    float getSk() const {
        if (m != 0) { // Return results if cached
            return m;
        }
        float s = 0;
        for (int i = N - 1; i >= 0; i--) {
            s = e[i] * (s + g[i]);
        }
        return s;
    }
};

struct Vertex {
    Vertex() = default;
    explicit Vertex(float _pdf) { pdf = _pdf; }
    Vertex(
        const Vector&   _wi,
        const Vector&   _wo,
        float           _pdf,
        const Spectrum& _weight,
        const Spectrum& _specAcc,
        float           _pdfAcc,
        float           _inLamda,
        float           _outLamda,
        float           _invPdf,
        float           _invPdfAcc
    ) {
        wi        = _wi;
        wo        = _wo;
        pdf       = _pdf;
        weight    = _weight;
        weightAcc = _specAcc;
        pdfAcc    = _pdfAcc;
        inLamda   = _inLamda;
        outLamda  = _outLamda;
        invPdf    = _invPdf;
        invPdfAcc = _invPdfAcc;
    }

    Vector   wi{};
    Vector   wo{};
    Spectrum weight;
    Spectrum weightAcc;

    float pdf{};
    float pdfAcc{};
    float invPdf{};
    float invPdfAcc{};
    float inLamda{};
    float outLamda{};
};

struct PathSample {
    explicit PathSample(int order) {
        count = 0;
        vList.resize(order);
    }

    void
    add(const Vector&   _wi,
        const Vector&   _wo,
        float           _pdf,
        const Spectrum& _weight,
        const Spectrum& _specAcc,
        float           _pdfAcc,
        float           inLamda,
        float           outLamda,
        float           invPdf,
        float           _invPdfAcc) {
        vList[count] = (Vertex(_wi, _wo, _pdf, _weight, _specAcc, _pdfAcc, inLamda, outLamda, invPdf, _invPdfAcc));
        count++;
    }

    void add(float _pdf) {
        vList[count] = (Vertex(_pdf));
        count++;
    }
    std::vector<Vertex> vList{};
    int                 count;
};

struct PathSimple {
    explicit PathSimple(int order) {
        count = 0;
        pdf.resize(order);
        invPdf.resize(order);
    }

    void add(float _pdf, float _invPdf) {
        pdf[count]    = (_pdf);
        invPdf[count] = (_invPdf);
        count++;
    }
    std::vector<float> pdf{};
    std::vector<float> invPdf{};
    int                count;
};

// From the released code of Heitz et al. 2016
struct RayInfo {
    // direction
    MicrofacetDistribution::EType type;
    Vector3                       w{};
    float                         theta{};
    float                         cosTheta{};
    float                         sinTheta{};
    float                         tanTheta{};
    float                         alpha{};
    float                         Lambda{};

    RayInfo() : type(MicrofacetDistribution::EType::EGGX) {}
    explicit RayInfo(MicrofacetDistribution::EType m) : type(m) {}
    RayInfo(MicrofacetDistribution::EType m, Vector3 w0, const float alpha_x, const float alpha_y) : type(m) {
        updateDirection(w0, alpha_x, alpha_y);
    }

    void updateDirection(const Vector3& w, const float alpha_x, const float alpha_y) {
        if (type == MicrofacetDistribution::EType::EGGX) {
            this->w                  = w;
            cosTheta                 = Frame::cosTheta(w);
            theta                    = acosf(cosTheta);
            sinTheta                 = sinf(theta);
            tanTheta                 = sinTheta / cosTheta;
            const float invSinTheta2 = 1.0f / (1.0f - cosTheta * cosTheta);
            const float cosPhi2      = w.x * w.x * invSinTheta2;
            const float sinPhi2      = w.y * w.y * invSinTheta2;
            alpha                    = sqrtf(cosPhi2 * alpha_x * alpha_x + sinPhi2 * alpha_y * alpha_y);
            // Lambda
            if (cosTheta > 0.9999f)
                Lambda = 0.0f;
            else if (cosTheta < -0.9999f)
                Lambda = 1.0f;
            else {
                const float a = 1.0f / tanTheta / alpha;
                Lambda        = 0.5f * (((a < 0) ? 1.0f : -1.0f) + sqrtf(1 + 1 / (a * a)));
            }
        } else {
            this->w                  = w;
            cosTheta                 = Frame::cosTheta(w);
            theta                    = acosf(cosTheta);
            sinTheta                 = sinf(theta);
            tanTheta                 = sinTheta / cosTheta;
            const float invSinTheta2 = 1.0f / (1.0f - cosTheta * cosTheta);
            const float cosPhi2      = w.x * w.x * invSinTheta2;
            const float sinPhi2      = w.y * w.y * invSinTheta2;
            alpha                    = sqrtf(cosPhi2 * alpha_x * alpha_x + sinPhi2 * alpha_y * alpha_y);
            // Lambda
            if (cosTheta > 0.9999f)
                Lambda = 0.0f;
            else if (cosTheta < -0.9999f)
                Lambda = 1.0f;
            else {
                const float a = 1.0f / tanTheta / alpha;
                Lambda        = abs(0.5f * ((float)erf(a) - 1.0f) + 0.28209479177f / a * expf(-a * a));
            }
        }
    }
};

class GMatrix {
public:
    int                width, height;
    std::vector<float> mat{};
    GMatrix(int width, int height) : width(width), height(height) { mat.resize(width * height); }
    float& operator()(int i, int j) { return mat[i * height + j]; }
    float  operator()(int i, int j) const { return mat[i * height + j]; }
    float& operator()(Point2i p) { return mat[p.x * height + p.y]; }
    float  operator()(Point2i p) const { return mat[p.x * height + p.y]; }
};

template <typename T>
T pow2(T a) {
    return a * a;
}

class RoughConductor : public BSDF {
public:
    explicit RoughConductor(const Properties& props) : BSDF(props) {
        ref<FileResolver> fResolver = Thread::getThread()->getFileResolver();

        m_specularReflectance = new ConstantSpectrumTexture(props.getSpectrum("specularReflectance", Spectrum(1.0f)));

        std::string materialName = props.getString("material", "Cu");

        Spectrum intEta, intK;
        if (boost::to_lower_copy(materialName) == "none") {
            intEta = Spectrum(0.0f);
            intK   = Spectrum(1.0f);
        } else {
            intEta.fromContinuousSpectrum(
                InterpolatedSpectrum(fResolver->resolve("data/ior/" + materialName + ".eta.spd"))
            );
            intK.fromContinuousSpectrum(InterpolatedSpectrum(fResolver->resolve("data/ior/" + materialName + ".k.spd"))
            );
        }

        Float extEta = lookupIOR(props, "extEta", "air");

        m_eta = props.getSpectrum("eta", intEta) / extEta;
        m_k   = props.getSpectrum("k", intK) / extEta;

        MicrofacetDistribution distr(props);
        m_type          = distr.getType();
        m_sampleVisible = distr.getSampleVisible();

        m_alphaU = new ConstantFloatTexture(distr.getAlphaU());
        if (distr.getAlphaU() == distr.getAlphaV())
            m_alphaV = m_alphaU;
        else
            m_alphaV = new ConstantFloatTexture(distr.getAlphaV());
        m_order   = props.getInteger("order", 10);
        m_rrDepth = props.getInteger("rrDepth", 3);

        m_bdpt = props.getBoolean("BDPT", false);
    }

    RoughConductor(Stream* stream, InstanceManager* manager) : BSDF(stream, manager) {
        m_type                = (MicrofacetDistribution::EType)stream->readUInt();
        m_sampleVisible       = stream->readBool();
        m_alphaU              = static_cast<Texture*>(manager->getInstance(stream));
        m_alphaV              = static_cast<Texture*>(manager->getInstance(stream));
        m_specularReflectance = static_cast<Texture*>(manager->getInstance(stream));
        m_eta                 = Spectrum(stream);
        m_k                   = Spectrum(stream);

        configure();
    }

    void serialize(Stream* stream, InstanceManager* manager) const override {
        BSDF::serialize(stream, manager);

        stream->writeUInt((uint32_t)m_type);
        stream->writeBool(m_sampleVisible);
        manager->serialize(stream, m_alphaU.get());
        manager->serialize(stream, m_alphaV.get());
        manager->serialize(stream, m_specularReflectance.get());
        m_eta.serialize(stream);
        m_k.serialize(stream);
    }

    void configure() override {
        unsigned int extraFlags = 0;
        if (m_alphaU != m_alphaV)
            extraFlags |= EAnisotropic;

        if (!m_alphaU->isConstant() || !m_alphaV->isConstant() || !m_specularReflectance->isConstant())
            extraFlags |= ESpatiallyVarying;

        m_components.clear();
        m_components.push_back(EGlossyReflection | EFrontSide | extraFlags);

        /* Verify the input parameters and fix them if necessary */
        m_specularReflectance = ensureEnergyConservation(m_specularReflectance, "specularReflectance", 1.0f);

        m_usesRayDifferentials = m_alphaU->usesRayDifferentials() || m_alphaV->usesRayDifferentials() ||
                                 m_specularReflectance->usesRayDifferentials();

        m_usesRayDifferentials = true;

        BSDF::configure();
    }

    static inline bool IsFiniteNumber(float x) {
        return (x <= std::numeric_limits<Float>::max() && x >= -std::numeric_limits<Float>::max());
    }

    static inline Vector reflect(const Vector& wi, const Normal& m) { return 2 * dot(wi, m) * Vector(m) - wi; }

    Spectrum eval(BSDFSamplingRecord& bRec, EMeasure measure) const override {
        /* Stop if this component was not requested */
        if (measure != ESolidAngle || Frame::cosTheta(bRec.wi) <= 0 || Frame::cosTheta(bRec.wo) <= 0 ||
            ((bRec.component != -1 && bRec.component != 0) || !(bRec.typeMask & EGlossyReflection)))
            return Spectrum(0.0f);
        auto pRec = bRec;

        float alpha_x = m_alphaU->eval(bRec.its).average();
        float alpha_y = m_alphaV->eval(bRec.its).average();

        return m_bdpt ? evalBDPT_new(pRec, m_order, measure, alpha_x, alpha_y)
                      : evalPT_new(pRec, m_order, measure, alpha_x, alpha_y);
    }

    float pdfWi(const Vector& wi, const Vector& wo, float inLamda, float alpha_x, float alpha_y) const {
        Vector                 m  = normalize(wo + wi);
        float                  G1 = 1.0f / (1 + inLamda);
        MicrofacetDistribution distr(m_type, alpha_x, alpha_y, true);
        float                  lmd = RayInfo(m_type, -wi, alpha_x, alpha_y).Lambda;
        float                  pdf = distr.eval(m) / (4 * abs(lmd) * abs(wi.z));

        return pdf;
    }

    inline Float pdfVisible(const Vector& wi, const Vector& m, float alpha_x, float alpha_y) const {
        if (Frame::cosTheta(wi) == 0)
            return 0.0f;

        float G1 = computeG1(wi, alpha_x, alpha_y);
        if (!IsFiniteNumber(G1))
            return 0.0f;

        MicrofacetDistribution distr(m_type, alpha_x, alpha_y, true);
        float                  lmd = RayInfo(m_type, -wi, alpha_x, alpha_y).Lambda;
        return absDot(wi, m) * distr.eval(m) / (abs(lmd) * abs(wi.z));
    }
    // single bounce
    Spectrum evalBounce(const Vector& wi, const Vector& wo, float alpha_x, float alpha_y) const {
        Spectrum result = computeD_F(wi, wo, alpha_x, alpha_y);
        if (result.isZero())
            return Spectrum(0.0f);

        float G  = computeG2(-wi, wo, alpha_x, alpha_y);
        result  *= G;

        return result;
    }
    float getInLambda(const Vector3& wi, float alpha_x, float alpha_y) const {
        RayInfo ray(m_type);
        ray.updateDirection(-wi, alpha_x, alpha_y);
        return ray.Lambda - 1;
    }

    float pdfVNDF(const Vector& wi, const Vector& wo, float alpha_x, float alpha_y) const {
        Normal                 m = normalize(wi + wo);
        MicrofacetDistribution distr(m_type, alpha_x, alpha_y, true);
        /* Sample M, the microfacet normal */
        // m = distr.sampleVisible(wi, rand);
        // pdf = pdfVisible(wi, m, alpha_x, alpha_y);
        float lmd = RayInfo(m_type, -wi, alpha_x, alpha_y).Lambda;
        float D   = distr.eval(m);
        return D / (4 * abs(lmd * wi.z));
    }
    // sample a subpath from one starting direction.
    void samplePath(
        const Vector& wi_,
        Sampler*      sampler,
        int           order,
        PathSample&   path,
        bool          back,
        float         alpha_x,
        float         alpha_y
    ) const {
        float    pdf       = 1.0f;
        float    invPdfAcc = 1.0f;
        float    pdfAcc    = 1.0f;
        Spectrum weightAcc(1.0f);
        Vector   wi      = wi_;
        float    inLamda = getInLambda(wi, alpha_x, alpha_y);

        for (int i = 0; i < order - 1; i++) {
            float currentPDF;
            // test
			Vector woNew = sampleVNDF(wi, alpha_x, alpha_y, Point2(generateRandomNumber(), generateRandomNumber()));
            currentPDF    = pdfVNDF(wi, woNew, alpha_x, alpha_y);
            pdf          *= currentPDF;

            if (!(wi + woNew).isZero() && pdf > 1e-4) {
                Spectrum weight;
                Vector   wo          = woNew;
                float    outLamda    = getLambda(wo, alpha_x, alpha_y);
                float    mapOutLamda = abs(outLamda + 1) - 1;
                if (back) {
                    wo = wi;
                    wi = woNew;

                    weight = computeD_F(wi, wo, alpha_x, alpha_y) / currentPDF;

                    weightAcc *= weight;

                    float forwardPdf = 1;

                    forwardPdf  = i == 0 ? 1.0f : pdfWi(wi, wo, mapOutLamda, alpha_x, alpha_y);
                    pdfAcc     *= forwardPdf;


                    path.add(wo, wi, forwardPdf, weight, weightAcc, pdfAcc, inLamda, outLamda, currentPDF, pdf);
                } else {
                    weight     = computeD_F(wi, wo, alpha_x, alpha_y) / currentPDF;
                    weightAcc *= weight;

                    float backPdf = 1;

                    wo = wi;
                    wi = woNew;

                    backPdf    = i == 0 ? 1 : pdfWi(wi, wo, mapOutLamda, alpha_x, alpha_y);
                    invPdfAcc *= backPdf;

                    // the order is always from the macro incmoing direction side to the other side.
                    path.add(wo, wi, currentPDF, weight, weightAcc, pdf, inLamda, outLamda, backPdf, invPdfAcc);
                }
                wi      = -woNew;
                inLamda = abs(outLamda) - 1; //
            } else {
                path.add(0.0);
                break;
            }

            if (m_rrDepth > -1 && i + 1 >= m_rrDepth) {
                Float q = std::min(weightAcc.max(), (Float)0.95f);

				if (generateRandomNumber() > q) {
                    path.add(0.0);
                    break;
                } else {
                    weightAcc /= q;
                }
            }
        }
    }

    float getLambda(const Vector3& w, float alpha_x, float alpha_y) const {
        RayInfo ray(m_type);
        ray.updateDirection(w, alpha_x, alpha_y);
        return copysign(ray.Lambda, w.z);
    }

    float computeG2(const Vector3& wi, const Vector& wo, float alpha_x, float alpha_y) const {
        return 1.0f / (abs(getLambda(wo, alpha_x, alpha_y)) + abs(getLambda(wi, alpha_x, alpha_y)));
    }

    inline float computeG1(const Vector3& wi, float alpha_x, float alpha_y) const {
        float lambda = getLambda(wi, alpha_x, alpha_y);
        float G11    = 1.0f / abs(1.0f + lambda);
        return G11;
    }

    // G2 for all the cases
    static float computePathG(const std::vector<RayInfo>& dirList, int begin, int end) {
        int upPoint = end;
        for (int i = begin + 1; i < end; ++i) {
            if (dirList[i - 1].w.z <= 0.0f && dirList[i].w.z >= 0.0f) {
                upPoint = i;
                break;
            }
        }
        return computePathG(dirList, begin, end, upPoint);
    }
    static float computePathG(const std::vector<RayInfo>& dirList, int begin, int end, int upPoint) {
        GMatrix gMatrix(upPoint - begin, end - upPoint + 1);
        Point2i p1(gMatrix.width - 1, gMatrix.height - 1);
        gMatrix(p1) = 1.0f / (dirList[p1.x].Lambda + dirList[end - p1.y].Lambda);
        while (!p1.isZero()) {
            if (p1.x > 0) {
                p1.x -= 1;
            } else {
                p1.y -= 1;
            }
            Point2i p2(p1);
            do {
                gMatrix(p2) = (((p2.x + 1 < gMatrix.width) ? gMatrix(p2.x + 1, p2.y) : 0) +
                               ((p2.y + 1 < gMatrix.height) ? gMatrix(p2.x, 1 + p2.y) : 0)) /
                              (dirList[p2.x].Lambda + dirList[end - p2.y].Lambda);
                p2 += Point2i(1, -1);
            } while (p2.x < gMatrix.width && p2.y >= 0);
        }
        return gMatrix(0, 0);
    }

    // vertex term
    inline Spectrum computeD_F(const Vector& wi, const Vector& wo, float alpha_x, float alpha_y) const {
		if ((wo + wi).isZero())
			return Spectrum(0.0f);

		/* Calculate the reflection half-vector */
		Vector H = normalize(wo + wi);
		Float D = MicrofacetDistribution(m_type, alpha_x, alpha_y).eval(H);
		if (D == 0)
			return Spectrum(0.0f);

		/* Fresnel factor */
		const Spectrum F = fresnelConductorExact(dot(wi, H), m_eta, m_k);
		return D * F / (4 * abs(wi.z));
    }

    // vertex term, exept the Jacobian term, the D wil be cancelled out in the
    // sampling, so do not include here.
    inline Spectrum computeF(const Vector& wi, const Vector& wo) const {
        Vector H = wo + wi;
        if (H.isZero())
            return Spectrum(0.0f);
        H                = normalize(H);
        const Spectrum F = fresnelConductorExact(dot(wi, H), m_eta, m_k);
        return F;
    }

    typedef Vector float3;

    static float3 sampleGGXVNDF(float3 Wi, float alpha_x, float alpha_y, float U1, float U2) {
        float3 Vh    = normalize(float3(alpha_x * Wi.x, alpha_y * Wi.y, Wi.z));
        float  lensq = Vh.x * Vh.x + Vh.y * Vh.y;
        float3 T1    = lensq > 0 ? float3(-Vh.y, Vh.x, 0) / sqrt(lensq) : float3(1, 0, 0);
        float3 T2    = cross(Vh, T1);
        float  r     = sqrt(U1);
        float  phi   = 2.0f * M_PI * U2;
        float  t1    = r * cos(phi);
        float  t2    = r * sin(phi);
        float  s     = 0.5f * (1.0f + Vh.z);
        t2           = (1.0f - s) * sqrt(1.0f - t1 * t1) + s * t2;
        float3 Nh    = t1 * T1 + t2 * T2 + Vh * sqrt(std::max(0.0f, 1.0f - t1 * t1 - t2 * t2));
        float3 Ne    = normalize(float3(alpha_x * Nh.x, alpha_y * Nh.y, std::max(0.0f, Nh.z)));
        return Ne;
    }

    // sample the d_i+1 for the incoming -d_i with VNDF sampling
    inline void sampleVNDF(BSDFSamplingRecord& bRec, float alpha_x, float alpha_y) const {
        Normal m;

        /* Sample M, the microfacet normal */
		Point2 rand(generateRandomNumber(), generateRandomNumber());
        if (m_type == MicrofacetDistribution::EGGX) {
            m = sampleGGXVNDF(bRec.wi, alpha_x, alpha_y, rand.x, rand.y);
        } else {
            m = MicrofacetDistribution(m_type, alpha_x, alpha_y).sampleVisible(bRec.wi, rand);
        }
        bRec.wo = reflect(bRec.wi, m);
    }

    // sample the d_i+1 for the incoming -d_i with VNDF sampling
    inline Vector sampleVNDF(Vector const& wi, float alpha_x, float alpha_y, Point2 rand) const {
        Normal m;
        if (m_type == MicrofacetDistribution::EGGX) {
            m = sampleGGXVNDF(wi, alpha_x, alpha_y, rand.x, rand.y);
        } else {
            m = MicrofacetDistribution(m_type, alpha_x, alpha_y).sampleVisible(wi, rand);
        }
        return reflect(wi, m);
    }

    Spectrum evalPT_new(BSDFSamplingRecord& bRec, int order, EMeasure measure, float alpha_x, float alpha_y) const {
        Vector   w0  = bRec.wi;
        Vector   woN = bRec.wo;
        Spectrum result(0.0f);
        Spectrum weight(1.0f);

        float invPDF = getLambda(-w0, alpha_x, alpha_y);

        SegmentTerm s(getLambda(woN, alpha_x, alpha_y));

        s.addBounce(invPDF);

        result += computeD_F(w0, woN, alpha_x, alpha_y) * s.getSk();

        for (int i = 1; i < order; ++i) {
            sampleVNDF(bRec, alpha_x, alpha_y);

            weight *= computeF(bRec.wi, bRec.wo);

            float lambda = getLambda(bRec.wo, alpha_x, alpha_y);

            s.addBounce(lambda);

            bRec.wi = -bRec.wo;

            float Sk = s.getSk();

            if (i >= m_rrDepth) {
                Float q = std::max(std::min(Sk, 0.95f), (Float)0.3f);
                if (generateRandomNumber() >= q)
                    break; // russian roulette
                weight /= q;
            }

            result += computeD_F(bRec.wi, woN, alpha_x, alpha_y) * weight * abs(invPDF) * Sk;

            invPDF *= lambda;
        }

        bRec.wi = w0;
        bRec.wo = woN;

        return result;
    }

    // sum all the pdf of the all the possible constructions
    static float computeWeightSum(const PathSimple& path, int order) {
        float pdfSum = 0.0f;
        float pdfPre = 1;

        for (int k = 1; k < order - 1; k++) {
            float pdfTemp = 1; // pdfPre;
            for (int j = 0; j < k; j++) {
                pdfTemp *= path.pdf[j];
            }
            for (int j = k + 1; j < order; j++) {
                pdfTemp *= path.invPdf[j];
            }
            pdfSum += pdfTemp;
        }
        return pdfSum;
    }

    Spectrum evalBDPT_new(BSDFSamplingRecord& bRec, int order, EMeasure measure, float alpha_x, float alpha_y) const {
        Vector w0  = bRec.wi;
        Vector woN = bRec.wo;

        Spectrum result(0.0f);

        PathSample pathForward(order);
        PathSample pathBack(order);

        // sample the two subpaths
        samplePath(w0, bRec.sampler, order, pathForward, false, alpha_x, alpha_y);
        samplePath(woN, bRec.sampler, order, pathBack, true, alpha_x, alpha_y);

        float outLamda = getInLambda(woN, alpha_x, alpha_y);
        float inLamda  = getInLambda(w0, alpha_x, alpha_y);

        PathSimple path(order);
        Vector     wi = bRec.wi;
        Vector     wo = bRec.wo;

        Vector tempWi, tempWo;

        // sample count from 1 to order - 1, and the last one is connection
        for (int k = 0; k < order; k++) {
            Spectrum resultPathk(1.0);
            float    pdfPathK = 1.0F;
            if (k != 0) {
                const Vertex& v = pathForward.vList[k - 1];
                if (v.pdf == 0.0)
                    break;
                resultPathk *= pathForward.vList[k - 1].weightAcc;
                pdfPathK     = pathForward.vList[k - 1].pdfAcc;
            }

			if (resultPathk.getLuminance() < THRESHOLD || pdfPathK < THRESHOLD)
				break;

            for (int i = 0; i < order - k; i++) {
                if (i != 0) {
                    const Vertex& vt = pathBack.vList[i - 1];
                    if (vt.pdf == 0.0)
                        break;
                }

                int currentOrder = k + i + 1;

                // k samples for forward path and j samples from back paths
                if (k == 0 && i == 0) {
                    result += evalBounce(wi, wo, alpha_x, alpha_y);
                    continue;
                }
                if (k + i > order - 1)
                    continue;

                path.count            = 0; // reset the path.
                float    pdfForward   = 1.0;
                float    pdfBackward  = 1.0f;
                Spectrum resultPath   = resultPathk;
                float    pdfPath      = pdfPathK;
                pdfForward           *= pdfPathK;

                if (k > 0)
                    pdfBackward *= pathForward.vList[k - 1].invPdfAcc;

                if (i > 0) {
                    resultPath *= pathBack.vList[i - 1].weightAcc;
                    pdfPath    *= pathBack.vList[i - 1].invPdfAcc;

                    pdfForward  *= pathBack.vList[i - 1].pdfAcc;
                    pdfBackward *= pathBack.vList[i - 1].invPdfAcc;
                }

				if (resultPath.getLuminance() < THRESHOLD || pdfPath < THRESHOLD)
					break;

                for (int j = 0; j < k; j++) {
                    const Vertex& v = pathForward.vList[j];
                    path.add(v.pdf, v.invPdf);
                }

                if (i == 0) {
                    // connect the kth to wo
                    const Vertex& v    = pathForward.vList[k - 1];
                    tempWi             = -v.wo;
                    tempWo             = woN;
                    float inLamdaTemp  = abs(v.outLamda) - 1;
                    resultPath        *= computeD_F(tempWi, tempWo, alpha_x, alpha_y);
					if (pdfBackward > THRESHOLD){
						float pdf = 1;
						float invPdf = pdfWi(woN, -v.wo, outLamda, alpha_x, alpha_y);

						pdfForward *= pdf;
						pdfBackward *= invPdf;
						path.add(pdf, invPdf);
					}
					else{
						path.add(1, 0.0f);
					}
                }

                if (k != 0 && i != 0) // th
                {
                    // we should connect two of them
                    const Vertex& v_s          = pathForward.vList[k - 1];
                    const Vertex& v_t          = pathBack.vList[i - 1];
                    Vector        tempWi       = -v_s.wo;
                    Vector        tempWo       = -v_t.wo;
                    float         inLamdaTemp  = abs(v_s.outLamda) - 1;
                    float         outLamdaTemp = abs(v_t.outLamda) - 1;

                    resultPath *= computeD_F(tempWi, tempWo, alpha_x, alpha_y);
                    float pdf   = 0;
					if (pdfForward > THRESHOLD){
						pdf         = pdfWi(tempWi, tempWo, inLamdaTemp, alpha_x, alpha_y);
						pdfForward *= pdf;
					}
                    float invPdf = 0;
					if (pdfForward > THRESHOLD){
						tempWo = -v_s.wo;
						tempWi = -v_t.wo;
						invPdf = pdfWi(tempWi, tempWo, outLamdaTemp, alpha_x, alpha_y);
						pdfBackward *= invPdf;
					}
                    path.add(pdf, invPdf);
                }

                if (k == 0) // this is the first one
                {
                    const Vertex& v             = pathBack.vList[i - 1];
                    Vector        tempWi        = w0;
                    Vector        tempWo        = -v.wo;
                    float         outLamdaTemp  = abs(v.outLamda) - 1;
                    resultPath                 *= computeD_F(tempWi, tempWo, alpha_x, alpha_y);
                    float invPdf                = 1;
                    pdfBackward                *= invPdf;

                    float pdf = 0;
					if (pdfForward > THRESHOLD){
						pdf = pdfWi(tempWi, tempWo, inLamda, alpha_x, alpha_y);
						pdfForward *= pdf;
					}
                    path.add(pdf, invPdf);
                }

                for (int j = i - 1; j >= 0; j--) {
                    const Vertex& v = pathBack.vList[j];
                    path.add(v.pdf, v.invPdf);
                }

				if (resultPath.getLuminance() < THRESHOLD || pdfPath < THRESHOLD)
					continue;

                float pdfSum = pdfForward + pdfBackward + computeWeightSum(path, currentOrder);

                // dirlist
                std::vector<RayInfo> dirList;

                dirList.push_back(RayInfo(m_type, -w0, alpha_x, alpha_y));
                for (int j = 0; j < k; ++j) {
                    dirList.push_back(RayInfo(m_type, pathForward.vList[j].wo, alpha_x, alpha_y));
                }
                for (int j = i - 1; j >= 0; --j) {
                    dirList.push_back(RayInfo(m_type, -pathBack.vList[j].wo, alpha_x, alpha_y));
                }
                dirList.push_back(RayInfo(m_type, woN, alpha_x, alpha_y));

                float GPath = computePathG(dirList, 0, currentOrder);

                result += pdfSum == 0.0 ? Spectrum(0.0f) : resultPath * GPath * pdfPath / pdfSum;
            }
        }
        return result.isValid() ? result : Spectrum(0.0f);
    }

    float evalBouncePdf(const BSDFSamplingRecord& bRec, float alpha_x, float alpha_y) const {
        /* Calculate the reflection half-vector */
        Vector H = normalize(bRec.wo + bRec.wi);
        if (H.z <= 0)
            return 0.0f;
        Float D;
        if (m_type == MicrofacetDistribution::EGGX) {
            float  a2 = alpha_x * alpha_y;
            Vector V  = Vector(alpha_y * H.x, alpha_x * H.y, a2 * H.z);
            float  S  = dot(V, V);
            D         = INV_PI * a2 * pow2(a2 / S);
        } else {
            auto distr = MicrofacetDistribution(m_type, alpha_x, alpha_y);
            D          = distr.eval(H);
        }
        float G = computeG1(bRec.wi, alpha_x, alpha_y);

        Float model = D * G / (4 * Frame::cosTheta(bRec.wi));
        return model;
    }

    static float Happorx(float u, float a) { return (1 + 2 * u) / (1 + 2 + sqrt(1 - a) * u); } // apporximate H function

    Float pdf(const BSDFSamplingRecord& bRec, EMeasure measure) const override {
        if (measure != ESolidAngle || Frame::cosTheta(bRec.wi) <= 0 || Frame::cosTheta(bRec.wo) <= 0 ||
            ((bRec.component != -1 && bRec.component != 0) || !(bRec.typeMask & EGlossyReflection)))
            return 0.0f;

        float alpha_x = m_alphaU->eval(bRec.its).average();
        float alpha_y = m_alphaV->eval(bRec.its).average();

        float alpha = (alpha_x + alpha_y) * 0.5f;

        alpha = std::min((alpha), 0.975f);

        float pdf = evalBouncePdf(bRec, alpha_x, alpha_y) +
                    (alpha)*INV_PI * 0.25f * (Happorx(bRec.wi.z, alpha) * Happorx(bRec.wo.z, alpha) - 1) /
                        (bRec.wi.z + bRec.wo.z);

        return pdf;
    }

    Spectrum sample(BSDFSamplingRecord& bRec, const Point2& sample) const override {
        float pdf;
        return this->sample(bRec, pdf, sample);
    }

    Spectrum sample(BSDFSamplingRecord& bRec, Float& pdf, const Point2& sample) const override {
        if (Frame::cosTheta(bRec.wi) <= 0 ||
            ((bRec.component != -1 && bRec.component != 0) || !(bRec.typeMask & EGlossyReflection)))
            return Spectrum(0.0f);

        float alpha_x = m_alphaU->eval(bRec.its).average();
        float alpha_y = m_alphaV->eval(bRec.its).average();

        bRec.eta              = 1.0f;
        bRec.sampledComponent = 0;
        bRec.sampledType      = EGlossyReflection;

        pdf           = 1.0f;
        int    bounce = 0;
        Vector w0     = bRec.wi;

        std::vector<RayInfo> dirList;
        dirList.resize(m_order + 1);
        dirList[0]      = RayInfo(m_type, -w0, alpha_x, alpha_y);
        Spectrum weight = Spectrum(1.0f);

        int upPoint = -1;

        while (true) {
            sampleVNDF(bRec, alpha_x, alpha_y);

            pdf /= dirList[bounce].Lambda;
            if (pdf < 1e-10f)
                return Spectrum(0.0f);

            ++bounce;

            if (upPoint < 0 && bRec.wo.z > 0) {
                upPoint = bounce;
            }

            dirList[bounce] = RayInfo(m_type, bRec.wo, alpha_x, alpha_y);

            Spectrum F  = computeF(bRec.wi, bRec.wo);
            weight     *= F;

            if (Frame::cosTheta(bRec.wo) <= 0) {
                // the ray can not leave the surface
                if (bounce == m_order) {
                    bRec.wo = Vector3(0, 0, 1);
                    return Spectrum(0.0f);
                }
            } else {
                // the ray leave the surface, since the maximum bounce has reached
                if (bounce == m_order) {
                    break;
                }
                float G1 = 1.0f / (1 + dirList[bounce].Lambda);
                // the ray continues the tracing with G1 as the probablity
                if (bRec.sampler->next1D() < G1) {
                    pdf *= G1;
                    break;
                } else {
                    pdf *= (1 - G1);
                }
            }
            // the ray starts the next bounce
            bRec.wi = -bRec.wo;
        }
        weight *= computePathG(dirList, 0, bounce, upPoint);

        if (pdf < 1e-7f)
            return Spectrum(0.0f);
        weight /= pdf;

        bRec.wi = w0;

        pdf = this->pdf(bRec, ESolidAngle);

        return weight;
    }

    void addChild(const std::string& name, ConfigurableObject* child) override {
        SLog(EDebug, "%s", name);

        if (child->getClass()->derivesFrom(MTS_CLASS(Texture))) {
            if (name == "alpha") {
                m_alphaU = m_alphaV = static_cast<Texture*>(child);
            } else if (name == "alphaU") {
                m_alphaU = static_cast<Texture*>(child);
            } else if (name == "alphaV") {
                m_alphaV = static_cast<Texture*>(child);
            } else if (name == "specularReflectance") {
                m_specularReflectance = static_cast<Texture*>(child);
            } else {
                BSDF::addChild(name, child);
            }
        } else {
            BSDF::addChild(name, child);
        }
    }

    Float getRoughness(const Intersection& its, int component) const override {
        return 0.5f * (m_alphaU->eval(its).average() + m_alphaV->eval(its).average());
    }

    std::string toString() const override {
        std::ostringstream oss;
        oss << "RoughConductor[" << endl
            << "  id = \"" << getID() << "\"," << endl
            << "  distribution = " << MicrofacetDistribution::distributionName(m_type) << "," << endl
            << "  sampleVisible = " << m_sampleVisible << "," << endl
            << "  alphaU = " << indent(m_alphaU->toString()) << "," << endl
            << "  alphaV = " << indent(m_alphaV->toString()) << "," << endl
            << "  specularReflectance = " << indent(m_specularReflectance->toString()) << "," << endl
            << "  eta = " << m_eta.toString() << "," << endl
            << "  k = " << m_k.toString() << endl
            << "]";
        return oss.str();
    }
    Float getPhongExponent(const Intersection& its) const override {
        MicrofacetDistribution distr(
            MicrofacetDistribution::EPhong,
            m_alphaU->eval(its).average(),
            m_alphaV->eval(its).average(),
            m_sampleVisible
        );
        return distr.getExponent();
    }
    Shader* createShader(Renderer* renderer) const override;

    MTS_DECLARE_CLASS()
private:
    MicrofacetDistribution::EType m_type;
    ref<Texture>                  m_specularReflectance;
    ref<Texture>                  m_alphaU, m_alphaV;
    bool                          m_sampleVisible;
    Spectrum                      m_eta, m_k;

    // for mbndf
    int  m_order{};   // the maximum bounces of the multiple scattering
    bool m_bdpt{};    // using the bidirectional estimator
    int  m_rrDepth{}; // the depth of the Russian Roulette
};

/**
 * GLSL port of the rough conductor shader. This version is much more
 * approximate -- it only supports the Ashikhmin-Shirley distribution,
 * does everything in RGB, and it uses the Schlick approximation to the
 * Fresnel reflectance of conductors. When the roughness is lower than
 * \alpha < 0.2, the shader clamps it to 0.2 so that it will still perform
 * reasonably well in a VPL-based preview.
 */
class RoughConductorShader : public Shader {
public:
    RoughConductorShader(
        Renderer*       renderer,
        const Texture*  specularReflectance,
        const Texture*  alphaU,
        const Texture*  alphaV,
        const Spectrum& eta,
        const Spectrum& k
    )
    : Shader(renderer, EBSDFShader), m_specularReflectance(specularReflectance), m_alphaU(alphaU), m_alphaV(alphaV) {
        m_specularReflectanceShader = renderer->registerShaderForResource(m_specularReflectance.get());
        m_alphaUShader              = renderer->registerShaderForResource(m_alphaU.get());
        m_alphaVShader              = renderer->registerShaderForResource(m_alphaV.get());

        /* Compute the reflectance at perpendicular incidence */
        m_R0 = fresnelConductorExact(1.0f, eta, k);
    }

    bool isComplete() const override {
        return m_specularReflectanceShader.get() != NULL && m_alphaUShader.get() != NULL &&
               m_alphaVShader.get() != NULL;
    }

    void putDependencies(std::vector<Shader*>& deps) override {
        deps.push_back(m_specularReflectanceShader.get());
        deps.push_back(m_alphaUShader.get());
        deps.push_back(m_alphaVShader.get());
    }

    void cleanup(Renderer* renderer) override {
        renderer->unregisterShaderForResource(m_specularReflectance.get());
        renderer->unregisterShaderForResource(m_alphaU.get());
        renderer->unregisterShaderForResource(m_alphaV.get());
    }

    void
    resolve(const GPUProgram* program, const std::string& evalName, std::vector<int>& parameterIDs) const override {
        parameterIDs.push_back(program->getParameterID(evalName + "_R0", false));
    }

    void bind(GPUProgram* program, const std::vector<int>& parameterIDs, int& textureUnitOffset) const override {
        program->setParameter(parameterIDs[0], m_R0);
    }

    void generateCode(std::ostringstream& oss, const std::string& evalName, const std::vector<std::string>& depNames)
        const override {
        oss << "uniform vec3 " << evalName << "_R0;" << endl
            << endl
            << "float " << evalName << "_D(vec3 m, float alphaU, float alphaV) {" << endl
            << "    float ct = cosTheta(m), ds = 1-ct*ct;" << endl
            << "    if (ds <= 0.0)" << endl
            << "        return 0.0f;" << endl
            << "    alphaU = 2 / (alphaU * alphaU) - 2;" << endl
            << "    alphaV = 2 / (alphaV * alphaV) - 2;" << endl
            << "    float exponent = (alphaU*m.x*m.x + alphaV*m.y*m.y)/ds;" << endl
            << "    return sqrt((alphaU+2) * (alphaV+2)) * 0.15915 * pow(ct, "
               "exponent);"
            << endl
            << "}" << endl
            << endl
            << "float " << evalName << "_G(vec3 m, vec3 wi, vec3 wo) {" << endl
            << "    if ((dot(wi, m) * cosTheta(wi)) <= 0 || " << endl
            << "        (dot(wo, m) * cosTheta(wo)) <= 0)" << endl
            << "        return 0.0;" << endl
            << "    float nDotM = cosTheta(m);" << endl
            << "    return min(1.0, min(" << endl
            << "        abs(2 * nDotM * cosTheta(wo) / dot(wo, m))," << endl
            << "        abs(2 * nDotM * cosTheta(wi) / dot(wi, m))));" << endl
            << "}" << endl
            << endl
            << "vec3 " << evalName << "_schlick(float ct) {" << endl
            << "    float ctSqr = ct*ct, ct5 = ctSqr*ctSqr*ct;" << endl
            << "    return " << evalName << "_R0 + (vec3(1.0) - " << evalName << "_R0) * ct5;" << endl
            << "}" << endl
            << endl
            << "vec3 " << evalName << "(vec2 uv, vec3 wi, vec3 wo) {" << endl
            << "   if (cosTheta(wi) <= 0 || cosTheta(wo) <= 0)" << endl
            << "    	return vec3(0.0);" << endl
            << "   vec3 H = normalize(wi + wo);" << endl
            << "   vec3 reflectance = " << depNames[0] << "(uv);" << endl
            << "   float alphaU = max(0.005, " << depNames[1] << "(uv).r);" << endl
            << "   float alphaV = max(0.005, " << depNames[2] << "(uv).r);" << endl
            << "   float D = " << evalName << "_D(H, alphaU, alphaV)"
            << ";" << endl
            << "   float G = " << evalName << "_G(H, wi, wo);" << endl
            << "   vec3 F = " << evalName << "_schlick(1-dot(wi, H));" << endl
            << "   return reflectance * F * (D * G / (4*cosTheta(wi)));" << endl
            << "}" << endl
            << endl
            << "vec3 " << evalName << "_diffuse(vec2 uv, vec3 wi, vec3 wo) {" << endl
            << "    if (cosTheta(wi) < 0.0 || cosTheta(wo) < 0.0)" << endl
            << "    	return vec3(0.0);" << endl
            << "    return " << evalName << "_R0 * inv_pi * inv_pi * cosTheta(wo);" << endl
            << "}" << endl;
    }
    MTS_DECLARE_CLASS()
private:
    ref<const Texture> m_specularReflectance;
    ref<const Texture> m_alphaU;
    ref<const Texture> m_alphaV;
    ref<Shader>        m_specularReflectanceShader;
    ref<Shader>        m_alphaUShader;
    ref<Shader>        m_alphaVShader;
    Spectrum           m_R0;
};

Shader* RoughConductor::createShader(Renderer* renderer) const {
    return new RoughConductorShader(renderer, m_specularReflectance.get(), m_alphaU.get(), m_alphaV.get(), m_eta, m_k);
}

MTS_IMPLEMENT_CLASS(RoughConductorShader, false, Shader)
MTS_IMPLEMENT_CLASS_S(RoughConductor, false, BSDF)
MTS_EXPORT_PLUGIN(RoughConductor, "Rough conductor BRDF");
MTS_NAMESPACE_END
