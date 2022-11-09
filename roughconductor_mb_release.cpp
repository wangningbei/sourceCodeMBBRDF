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

#include <mitsuba/core/fresolver.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/hw/basicshader.h>
#include "microfacet.h"
#include "ior.h"
#define MTS_OPENMP 1
#if defined(MTS_OPENMP)
#include <omp.h>
#endif
#include <mitsuba/core/warp.h>

#define OPTIMIZE_PERFORMANCE 0
#define PI_DIVIDE_180 0.0174532922
#define INV_2_SQRT_M_PI	0.28209479177387814347f /* 0.5/sqrt(pi) */
#define THRESHOLD 1e-4
#define MAX_VERTEX 10
#define FRESNEL 1

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
*           \item \code{ggx}: The GGX \cite{Walter07Microfacet} distribution (also known as
*               Trowbridge-Reitz \cite{Trowbridge19975Average} distribution)
*               was designed to better approximate the long tails observed in measurements
*               of ground surfaces, which are not modeled by the Beckmann distribution.
*           \vspace{-1.5mm}
*           \item \code{phong}: Anisotropic Phong distribution by
*              Ashikhmin and Shirley \cite{Ashikhmin2005Anisotropic}.
*              In most cases, the \code{ggx} and \code{beckmann} distributions
*              should be preferred, since they provide better importance sampling
*              and accurate shadowing/masking computations.
*              \vspace{-4mm}
*       \end{enumerate}
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
*     \parameter{eta, k}{\Spectrum}{Real and imaginary components of the material's index of
*             refraction \default{based on the value of \texttt{material}}}
*     \parameter{extEta}{\Float\Or\String}{
*           Real-valued index of refraction of the surrounding dielectric,
*           or a material name of a dielectric \default{\code{air}}
*     }
*     \parameter{sampleVisible}{\Boolean}{
*         Enables a sampling technique proposed by Heitz and D'Eon~\cite{Heitz1014Importance},
*         which focuses computation on the visible parts of the microfacet normal
*         distribution, considerably reducing variance in some cases.
*         \default{\code{true}, i.e. use visible normal sampling}
*     }
*     \parameter{specular\showbreak Reflectance}{\Spectrum\Or\Texture}{Optional
*         factor that can be used to modulate the specular reflection component. Note
*         that for physical realism, this parameter should never be touched. \default{1.0}}
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
* This plugin is essentially the ``roughened'' equivalent of the (smooth) plugin
* \pluginref{conductor}. For very low values of $\alpha$, the two will
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
* \begin{xml}[caption={A material definition for brushed aluminium}, label=lst:roughconductor-aluminium]
* <bsdf type="roughconductor">
*     <string name="material" value="Al"/>
*     <string name="distribution" value="phong"/>
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



struct PathSimple{
	PathSimple(){ count = 0; }

	void add(float _pdf, float _invPdf){
		pdf[count] = _pdf;
		invPdf[count] = _invPdf;
		count++;
	}
	float pdf[MAX_VERTEX];
	float invPdf[MAX_VERTEX];
	int count;
};


struct Vertex{
	Vertex(){	}
	Vertex(float _pdf){ pdf = _pdf; }
	Vertex(const Vector &_wi, const Vector &_wo, float _pdf, const Spectrum &_weight,
		const Spectrum& _specAcc, float _pdfAcc,
		float _inLamda, float _outLamda, float _invPdf, float _invPdfAcc)
	{
		wi = _wi;
		wo = _wo;
		pdf = _pdf;
		weight = _weight;
		weightAcc = _specAcc;
		pdfAcc = _pdfAcc;
		inLamda = _inLamda;
		outLamda = _outLamda;
		invPdf = _invPdf;
		invPdfAcc = _invPdfAcc;
	}

	Vector wi;
	Vector wo;
	Spectrum weight;
	Spectrum weightAcc;

	float pdf;
	float pdfAcc;
	float invPdf;
	float invPdfAcc;
	float inLamda;
	float outLamda;
};

struct PathSample{
	PathSample(){ count = 0; }

	void add(const Vector &_wi, const Vector &_wo, float _pdf,
		const Spectrum &_weight,
		const Spectrum& _specAcc, float _pdfAcc,
		float inLamda, float outLamda, float invPdf, float _invPdfAcc)
	{
		vList[count] = Vertex(_wi, _wo, _pdf, _weight, _specAcc, _pdfAcc, inLamda, outLamda, invPdf, _invPdfAcc);
		count++;
	}

	void add(float _pdf){
		vList[count] = Vertex(_pdf);
		count++;
	}
	Vertex vList[MAX_VERTEX];
	int count;
};

//From the released code of Heitz et al. 2016
struct RayInfoBeckmann
{
	// direction
	Vector3 w;
	float theta;
	float cosTheta;
	float sinTheta;
	float tanTheta;
	float alpha;
	float Lambda;

	void updateDirection(const Vector3& w, const float alpha_x, const float alpha_y)
	{
		this->w = w;
		theta = acosf(w.z);
		cosTheta = w.z;
		sinTheta = sinf(theta);
		tanTheta = sinTheta / cosTheta;
		const float invSinTheta2 = 1.0f / (1.0f - w.z*w.z);
		const float cosPhi2 = w.x*w.x*invSinTheta2;
		const float sinPhi2 = w.y*w.y*invSinTheta2;
		alpha = sqrtf(cosPhi2*alpha_x*alpha_x + sinPhi2*alpha_y*alpha_y);
		// Lambda
		if (w.z > 0.9999f)
			Lambda = 0.0f;
		else if (w.z < -0.9999f)
			Lambda = -1.0f;
		else
		{
			const float a = 1.0f / tanTheta / alpha;
			//Lambda = 0.5f*(-1.0f + ((a>0)?1.0f:-1.0f) * sqrtf(1 + 1/(a*a)));
			Lambda = 0.5f*((float)erf(a) - 1.0f) + INV_2_SQRT_M_PI / a * expf(-a*a);

		}
	}

	// height
	float h;
	float C1;
	float G1;

	void updateHeight(const float& h)
	{
		this->h = h;
		C1 = std::min(1.0f, std::max(0.0f, 0.5f*(h + 1.0f)));

		if (this->w.z > 0.9999f)
			G1 = 1.0f;
		else if (this->w.z <= 0.0f)
			G1 = 0.0f;
		else
			G1 = powf(this->C1, this->Lambda);
	}
};

//From the released code of Heitz et al. 2016
struct RayInfoGGX
{
	// direction
	Vector3 w;
	float theta;
	float cosTheta;
	float sinTheta;
	float tanTheta;
	float alpha;
	float Lambda;

	void updateDirection(const Vector3& w, const float alpha_x, const float alpha_y)
	{
		this->w = w;
		theta = acosf(w.z);
		cosTheta = w.z;
		sinTheta = sinf(theta);
		tanTheta = sinTheta / cosTheta;
		const float invSinTheta2 = 1.0f / (1.0f - w.z*w.z);
		const float cosPhi2 = w.x*w.x*invSinTheta2;
		const float sinPhi2 = w.y*w.y*invSinTheta2;
		alpha = sqrtf(cosPhi2*alpha_x*alpha_x + sinPhi2*alpha_y*alpha_y);
		// Lambda
		if (w.z > 0.9999f)
			Lambda = 0.0f;
		else if (w.z < -0.9999f)
			Lambda = -1.0f;
		else
		{
			const float a = 1.0f / tanTheta / alpha;
			Lambda = 0.5f*(-1.0f + ((a>0) ? 1.0f : -1.0f) * sqrtf(1 + 1 / (a*a)));
		}
	}

	// height
	float h;
	float C1;
	float G1;

	void updateHeight(const float& h)
	{
		this->h = h;
		C1 = std::min(1.0f, std::max(0.0f, 0.5f*(h + 1.0f)));

		if (this->w.z > 0.9999f)
			G1 = 1.0f;
		else if (this->w.z <= 0.0f)
			G1 = 0.0f;
		else
			G1 = powf(this->C1, this->Lambda);
	}
};

class RoughConductor : public BSDF {
public:
	RoughConductor(const Properties &props) : BSDF(props) {
		ref<FileResolver> fResolver = Thread::getThread()->getFileResolver();

		m_specularReflectance = new ConstantSpectrumTexture(
			props.getSpectrum("specularReflectance", Spectrum(1.0f)));

		std::string materialName = props.getString("material", "Cu");

		Spectrum intEta, intK;
		if (boost::to_lower_copy(materialName) == "none") {
			intEta = Spectrum(0.0f);
			intK = Spectrum(1.0f);
		}
		else {
			intEta.fromContinuousSpectrum(InterpolatedSpectrum(
				fResolver->resolve("data/ior/" + materialName + ".eta.spd")));
			intK.fromContinuousSpectrum(InterpolatedSpectrum(
				fResolver->resolve("data/ior/" + materialName + ".k.spd")));
		}

		Float extEta = lookupIOR(props, "extEta", "air");

		m_eta = props.getSpectrum("eta", intEta) / extEta;
		m_k = props.getSpectrum("k", intK) / extEta;

		MicrofacetDistribution distr(props);
		m_type = distr.getType();
		m_sampleVisible = distr.getSampleVisible();

		m_alphaU = new ConstantFloatTexture(distr.getAlphaU());
		if (distr.getAlphaU() == distr.getAlphaV())
			m_alphaV = m_alphaU;
		else
			m_alphaV = new ConstantFloatTexture(distr.getAlphaV());
		m_order = props.getInteger("order", 1);
		m_rrDepth = props.getInteger("rrDepth", 3);

		m_bdpt = props.getBoolean("BDPT", false);
		m_hCorrelated = props.getBoolean("hCorrelated", false);
		Intersection its;
		m_alpha_x = m_alphaU->eval(its).average();
		m_alpha_y = m_alphaV->eval(its).average();
		m_distr = new MicrofacetDistribution(m_type, m_alpha_x, m_alpha_y, true);
	}

	RoughConductor(Stream *stream, InstanceManager *manager)
		: BSDF(stream, manager) {
		m_type = (MicrofacetDistribution::EType) stream->readUInt();
		m_sampleVisible = stream->readBool();
		m_alphaU = static_cast<Texture *>(manager->getInstance(stream));
		m_alphaV = static_cast<Texture *>(manager->getInstance(stream));
		m_specularReflectance = static_cast<Texture *>(manager->getInstance(stream));
		m_eta = Spectrum(stream);
		m_k = Spectrum(stream);

		configure();
	}

	~RoughConductor()
	{
		delete m_distr;
	}

	void serialize(Stream *stream, InstanceManager *manager) const {
		BSDF::serialize(stream, manager);

		stream->writeUInt((uint32_t)m_type);
		stream->writeBool(m_sampleVisible);
		manager->serialize(stream, m_alphaU.get());
		manager->serialize(stream, m_alphaV.get());
		manager->serialize(stream, m_specularReflectance.get());
		m_eta.serialize(stream);
		m_k.serialize(stream);
	}

	void configure() {
		unsigned int extraFlags = 0;
		if (m_alphaU != m_alphaV)
			extraFlags |= EAnisotropic;

		if (!m_alphaU->isConstant() || !m_alphaV->isConstant() ||
			!m_specularReflectance->isConstant())
			extraFlags |= ESpatiallyVarying;

		m_components.clear();
		m_components.push_back(EGlossyReflection | EFrontSide | extraFlags);

		/* Verify the input parameters and fix them if necessary */
		m_specularReflectance = ensureEnergyConservation(
			m_specularReflectance, "specularReflectance", 1.0f);

		m_usesRayDifferentials =
			m_alphaU->usesRayDifferentials() ||
			m_alphaV->usesRayDifferentials() ||
			m_specularReflectance->usesRayDifferentials();

		m_usesRayDifferentials = true;

		BSDF::configure();
	}

	
	/* some helper functions */
	inline Vector reflect(const Vector &wi, const Normal &m) const {
		return 2 * dot(wi, m) * Vector(m) - wi;
	}

	inline bool IsFiniteNumber(float x) const
	{
		return (x <= std::numeric_limits<Float>::max() && x >= -std::numeric_limits<Float>::max());
	}

	Spectrum eval(BSDFSamplingRecord &bRec, EMeasure measure) const{
	
		return  m_bdpt ? evalBDPT(bRec, m_order, measure) : evalPT(bRec, m_order, measure);
	}

	void getLambda(const Vector3 &wi, const Vector &wo, 
		float &inLambda, float &outLambda) const
	{
		if (m_type == MicrofacetDistribution::EBeckmann)
		{
			RayInfoBeckmann ray_shadowing;
			ray_shadowing.updateDirection(wo, m_alpha_x, m_alpha_y);

			RayInfoBeckmann ray;
			ray.updateDirection(wi, m_alpha_x, m_alpha_y);

			inLambda = ray.Lambda;
			outLambda = ray_shadowing.Lambda;
		}
		else
		{
			RayInfoGGX ray_shadowing;
			ray_shadowing.updateDirection(wo, m_alpha_x, m_alpha_y);

			RayInfoGGX ray;
			ray.updateDirection(wi, m_alpha_x, m_alpha_y);

			inLambda = ray.Lambda;
			outLambda = ray_shadowing.Lambda;
		}
	}

	float getInLambda(const Vector3 &wi) const
	{
		if (m_type == MicrofacetDistribution::EBeckmann)
		{
			RayInfoBeckmann ray;
			ray.updateDirection(-wi, m_alpha_x, m_alpha_y);
			return abs(ray.Lambda) - 1;
		}
		else
		{
			RayInfoGGX ray;
			ray.updateDirection(-wi, m_alpha_x, m_alpha_y);
			return abs(ray.Lambda) - 1;
		}
	}

	float getLambda(const Vector3 &wo) const
	{
		if (m_type == MicrofacetDistribution::EBeckmann)
		{
			RayInfoBeckmann ray;
			ray.updateDirection(wo, m_alpha_x, m_alpha_y);
			return (ray.Lambda);
		}
		else
		{
			RayInfoGGX ray;
			ray.updateDirection(wo, m_alpha_x, m_alpha_y);
			return (ray.Lambda);
		}
	}

	//height-correlated G2 for the middle bounce
	float computeG2_cor_middle(const Vector3 &wi, const Vector &wo) const
	{
		float inLambda, outLambda;
		getLambda(wi, wo, inLambda, outLambda);

		const float Gtemp = 1.0f / (abs(1.0f + inLambda) + outLambda);
		const float Gtemp2 = 1.0f / (abs(1.0f + inLambda));
		float  G = Gtemp2 - Gtemp;
		return G;
	}
	//height-correlated G2 for the last bounce
	float computeG2_cor_last(const Vector3 &wi, const Vector &wo) const
	{
		float inLambda, outLambda;
		getLambda(wi, wo, inLambda, outLambda);
		float temp = (abs(1.0f + inLambda) + outLambda);
		float G = abs(temp) < 1e-10 ? 0.0 : 1.0f / temp;
		return  G;
	}
	//height-uncorrelated G2 for the last bounce
	float computeG2_uncor_last(const Vector3 &wi, const Vector &wo) const
	{
		float inLambda, outLambda;
		getLambda(wi, wo, inLambda, outLambda);

		float G11 = 1.0f / abs(1.0f + inLambda);
		float G12 = 1.0f / abs(1.0f + outLambda);
		float G = G11 * G12;
		return G;
	}

	//G2 for the last bounce (performance optimized version)
	float computeG2_middle_opt(BSDFSamplingRecord &bRec, float inLamda, float outLamda)const
	{
		float G;

		if (bRec.wo.z < 0)
		{
			G = 1.0 / (1 + inLamda);
		}
		else
		{
			if (m_hCorrelated)
			{
				const float Gtemp = 1.0f / (1.0f + inLamda + outLamda);
				const float Gtemp2 = 1.0f / (1.0f + inLamda);
				G = Gtemp2 - Gtemp;
			}
			else
			{
				float G11 = 1.0 / (1 + inLamda);
				float G12 = 1.0f / (1 + outLamda);
				G = G11 * (1 - G12);
			}
		}
		return G;
	}

	inline float computeG1(const Vector3 &wi) const
	{
		float lambda = getLambda(wi);
		float G11 = 1.0f / abs(1.0f + lambda);
		return G11;
	}

	//G2 for all the cases
	float computeG(const Vector &wi, const Vector &wo, bool outShadow)const
	{
		float G;

		if (outShadow)
		{
			G = m_hCorrelated ? computeG2_cor_last(wi, wo) :
				computeG2_uncor_last(wi, wo);
		}
		else
		{
			if (wo.z < 0)
			{
				G = computeG1(wi);
			}
			else
			{
				if (m_hCorrelated)
				{
					G = computeG2_cor_middle(wi, wo);
				}
				else
				{
					float G11 = computeG1(wi); 
					float G12 = computeG1(wo); 
					G = G11 * (1 - G12);
				}
			}
		}

		return G;
	}

	//vertex term, exept the Jacobian term
	inline Spectrum computeD_F(const Vector &wi, const Vector &wo) const
	{
		if ((wo + wi).isZero())
			return Spectrum(0.0f);

		/* Calculate the reflection half-vector */
		Vector H = normalize(wo + wi);
		Float D = m_distr->eval(H);
		if (D == 0)
			return Spectrum(0.0f);

		/* Fresnel factor */
		const Spectrum F =
#if	FRESNEL
			fresnelConductorExact(dot(wi, H), m_eta, m_k);
#endif
		return D*F;
	}

	//vertex term, exept the Jacobian term, the D wil be cancelled out in the sampling, so do not include here.
	inline Spectrum computeD_F_withoutD(const Vector &wi, const Vector &wo) const
	{
		if ((wo + wi).isZero())
			return Spectrum(0.0f);
		/* Calculate the reflection half-vector */
		Vector H = normalize(wo + wi);
		Float D = m_distr->eval(H);

		if (D == 0)
			return Spectrum(0.0f);

		/* Fresnel factor */
		const Spectrum F =
#if	FRESNEL
			fresnelConductorExact(dot(wi, H), m_eta, m_k);
#endif
		return F;
	}

	Spectrum evalBounce_opt(BSDFSamplingRecord &bRec, bool outShadow, float inLamda, float outLamda) const
	{
		Spectrum result = computeD_F(bRec.wi, bRec.wo);
		if (result.isZero()) return Spectrum(0.0f);

		float G;
		if (outShadow)
		{
			G = m_hCorrelated ? 1.0 / (1.0f + inLamda + outLamda) : 1.0f / ((1.0f + inLamda)*(1.0f + outLamda));
		}
		else
		{
			G = computeG2_middle_opt(bRec, inLamda, outLamda);
		}

		result *= G / abs(4.0f * Frame::cosTheta(bRec.wi));
		return result;
	}

	Spectrum evalBounce(const Vector &wi, const Vector &wo, bool outShadow) const
	{
		Spectrum result = computeD_F(wi, wo);
		if (result.isZero()) return Spectrum(0.0f);

		float G = computeG(wi, wo, outShadow);
		result *= G / (4 * abs(wi.z));

		return result;
	}

	inline Float pdfVisible(const Vector &wi, const Vector &m) const {
		if (Frame::cosTheta(wi) == 0)
			return 0.0f;

		float G1 = computeG1(wi);
		if (!IsFiniteNumber(G1))
			return 0.0f;

		return  G1* absDot(wi, m) * m_distr->eval(m) / std::abs(Frame::cosTheta(wi));
	}

	inline Vector sampleWi(const Vector &wi, Point2 rand, float &pdf) const
	{
		Normal m;
		/* Sample M, the microfacet normal */
		m = m_distr->sampleVisible(wi, rand);
		pdf = pdfVisible(wi, m);

		Vector wo = reflect(wi, m);
		pdf /= (4.0f * absDot(wo, m));
		return wo;
	}


	float pdfWi(BSDFSamplingRecord &bRec) const
	{
		Vector H = normalize(bRec.wo + bRec.wi);
		float pdf = pdfVisible(bRec.wi, H);

		return pdf / (4.0f * absDot(bRec.wo, H));
	}

	float pdfWi(const Vector &wi, const Vector &wo, float inLamda) const
	{
		Vector m = normalize(wo + wi);
		float G1 = 1.0f / (1 + inLamda);
		float pdf = G1* m_distr->eval(m) / (4 * std::abs(Frame::cosTheta(wi)));

		return pdf;
	}

	float pdfWi(BSDFSamplingRecord &bRec, float inLamda) const
	{
		Vector m = normalize(bRec.wo + bRec.wi);
		float G1 = 1.0f / (1 + inLamda);
		float pdf = G1* m_distr->eval(m) / (4 * std::abs(Frame::cosTheta(bRec.wi)));

		return pdf;
	}

	//the outLamda chould be less than 0, if wo is less than 0. Fix this later
	Spectrum evalBounceLast_opt(const Vector &wi, const Vector &wo, float inLamda, float outLamda) const
	{
		Spectrum result = computeD_F(wi, wo);
		if(result.isZero()) return Spectrum(0.0f);

		float G;
		if (m_hCorrelated)
		{
			float temp = (1.0f + (inLamda)+(outLamda));
			G = abs(temp) < 1e-20 ? 0.0 : 1.0f / temp;
		}
		else
		{
			G = 1.0f / ((1.0f + (inLamda))*(1.0f + (outLamda)));
		}

		if (!IsFiniteNumber(G))
			return Spectrum(0.0f);

		result *= G / (4 * abs(wi.z));

		return result;
	}

Spectrum evalBounceLast(const Vector &wi, const Vector &wo) const
{
	Spectrum result = computeD_F(wi, wo);
	if (result.isZero()) return Spectrum(0.0f);

	float G = computeG(wi, wo, true); 

	if (!IsFiniteNumber(G))
		return Spectrum(0.0f);

	result *= G / (4 * abs(wi.z));
	return result;
}

Spectrum evalBounceSample(const Vector &wi, const Vector &wo ) const
{
	Spectrum result = computeD_F_withoutD(wi, wo);
	if (result.isZero()) return Spectrum(0.0f);

	float G = computeG(wi, wo, false);

	if (!IsFiniteNumber(G))
		return Spectrum(0.0f);

	if (G == 0.0)
		return Spectrum(0.0f);

	//the Jacbian term is alrady included in the sample
	result *= G;

	return result;
}

Spectrum evalBounceSample_opt(BSDFSamplingRecord &bRec, float inLamda, float outLamda) const
{
	Spectrum result = computeD_F_withoutD(bRec.wi, bRec.wo);
	if (result.isZero()) return Spectrum(0.0f);

	float G = computeG2_middle_opt(bRec, inLamda, outLamda);

	if (!IsFiniteNumber(G))
		return Spectrum(0.0f);

	if (G == 0.0)
		return Spectrum(0.0f);

	result *= G;
	return result;

}

Spectrum evalBounceSampleMiddle(const Vector &wi, const Vector &wo, float inLamda, float outLamda) const
{
	Spectrum result = computeD_F(wi, wo);
	if (result.isZero()) return Spectrum(0.0f);

#if OPTIMIZE_PERFORMANCE 
	float G = computeGSampleInOut(bRec, inLamda, outLamda);
#else
	float G = computeG(wi,wo, false);
#endif

	if (!IsFiniteNumber(G))
		return Spectrum(0.0f);

	if (G == 0.0)
		return Spectrum(0.0f);

	result *= G / (4 * abs(wi.z));

	return result;

}

//sample the d_i+1 for the incoming -d_i with VNDF sampling
inline void sampleVNDF(BSDFSamplingRecord &bRec, float &pdf) const
{
	Normal m;

	/* Sample M, the microfacet normal */
	Point2 rand = bRec.sampler->next2D();
	m = m_distr->sampleVisible(bRec.wi, rand);

	float D = m_distr->eval(m);

	if (D == 0.0)
	{
		pdf = 0.0;
		return;
	}
#if OPTIMIZE_PERFORMANCE
	float G1 = 1.0 / (1 + inLamda); //// 
#else
	float G1 = computeG1(bRec.wi);
#endif
	if (!IsFiniteNumber(G1))
		G1 = 0.0f;

	pdf = G1 * absDot(bRec.wi, m) * m_distr->eval(m) / std::abs(Frame::cosTheta(bRec.wi));
	if (pdf == 0.0)
	{
		return;
	}
	pdf = G1;

	bRec.wo = reflect(bRec.wi, m);
}

/*our unidirectional estimator for the path integral*/
Spectrum evalPT(BSDFSamplingRecord &bRec,
	int order, EMeasure measure) const
{
	/* Stop if this component was not requested */
	if (Frame::cosTheta(bRec.wi) <= 0 ||
		Frame::cosTheta(bRec.wo) <= 0)
		return Spectrum(0.0f);

	Vector wi = bRec.wi;
	Vector wo = bRec.wo;

	float pdf = 1;
	Vector w0 = bRec.wi;
	Vector woN = bRec.wo;
	Spectrum result(0.0f);

	// for the single scattering
#if OPTIMIZE_PERFORMANCE
	float outLamda = getLambda(woN);
	float inLamda = getInLambda(w0);
	float readyOutLamda = abs(outLamda + 1) - 1;
	result += evalBounceLast_opt(bRec, inLamda, readyOutLamda);
#else
	result += evalBounceLast(wi, wo);
#endif

	Spectrum weight = Spectrum(1.0f);

	for (int i = 0; i < order - 1; i++)
	{
		float currentPDF = 1.0f;
		//bRec.wo = sampleVNDF(bRec.wi, bRec.sampler->next2D(), currentPDF);
		sampleVNDF(bRec, currentPDF);
		if (currentPDF < 1e-10) break;

		pdf *= currentPDF;

#if OPTIMIZE_PERFORMANCE
		float newOutLamda = getLambda(bRec.wo);
		float readyNewOutLamda = abs(newOutLamda + 1) - 1;
		weight *= evalBounceSample_opt(bRec, inLamda, readyNewOutLamda);
#else
		weight *= evalBounceSample(bRec.wi, bRec.wo); 
#endif
		if (pdf <= 1e-4) break;

		//next event estimation
		bRec.wi = -bRec.wo;
		bRec.wo = woN;

#if OPTIMIZE_PERFORMANCE
		inLamda = abs(newOutLamda) - 1;
		Spectrum currentWight = weight * evalBounceLast_opt(bRec, inLamda, readyOutLamda);
#else
		Spectrum currentWight = weight * evalBounceLast(bRec.wi, bRec.wo);
#endif
		result += currentWight / pdf;

		if (i+1 >= m_rrDepth) {

			Float q = std::min(currentWight.max(), (Float) 0.95f);
			if (bRec.sampler->next1D() >= q)
				break;
			weight /= q;
		}
	}

	bRec.wi = w0;
	bRec.wo = woN;
	return result;
}

//sample a subpath from one starting direction.
void samplePath(const Vector &wi_, Sampler  *sampler, int order, PathSample &path, bool back)const{

	float pdf = 1.0f;
	float invPdfAcc = 1.0f;
	float pdfAcc = 1.0f;
	Spectrum weightAcc(1.0f);
	Vector wi = wi_;
	float inLamda = getInLambda(wi);
	
	for (int i = 0; i < order - 1; i++)
	{
		float currentPDF;
		float simplePDF;
		Point2 rand = sampler->next2D();
		Vector woNew = sampleWi(wi, rand, currentPDF);
		pdf *= currentPDF;

		if (!(wi + woNew).isZero() && pdf > 1e-4)
		{
			Spectrum weight;
			Vector wo = woNew;
			float outLamda = getLambda(wo);
			float mapOutLamda = abs(outLamda + 1) - 1;
			if (back)
			{
				wo = wi;
				wi = woNew;
#if OPTIMIZE_PERFORMANCE
				weight = evalBounce_opt(bRec, i == 0, mapOutLamda, inLamda) / currentPDF;
#else
				weight = evalBounce(wi, wo, i == 0) / currentPDF;
#endif
				weightAcc *= weight;

				if (weightAcc.getLuminance() <= 1e-4)
				{
					path.add(0.0);
					break;
				}
				float forwardPdf = 1;
				if (pdfAcc > 1e-2)
				{
					forwardPdf = i == 0 ? 1.0 : pdfWi(wi, wo, mapOutLamda);
					pdfAcc *= forwardPdf;
				}

				path.add(wo, wi, forwardPdf, weight, weightAcc, pdfAcc, inLamda, outLamda, currentPDF, pdf);
			}
			else
			{
				weight = evalBounce(wi, wo, false) / currentPDF;
				weightAcc *= weight;
				if (weightAcc.getLuminance() <= 1e-4)
				{
					path.add(0.0);
					break;
				}
				float backPdf = 1;
				if (invPdfAcc > 1e-2)
				{
					wo = wi;
					wi = woNew;

					backPdf = i == 0 ? 1 : pdfWi(wi, wo, mapOutLamda);
					invPdfAcc *= backPdf;
				}
				//the order is always from the macro incmoing direction side to the other side.
				path.add(wo, wi, currentPDF, weight, weightAcc, pdf, inLamda, outLamda, backPdf, invPdfAcc);

			}
			wi = -woNew;
			inLamda = abs(outLamda) - 1;// 
		}
		else
		{
			path.add(0.0);
			break;
		}

		if (m_rrDepth > -1 && i + 1 >= m_rrDepth) {
			Float q = std::min(weightAcc.max(), (Float) 0.95f);

			if (sampler->next1D() > q) {
				path.add(0.0);
				break;
			}
			else {
				weightAcc /= q;
			}
		}

	}
}

//sum all the pdf of the all the possible constructions
float computeWeightSum(const PathSimple &path, int order) const
{
	float pdfSum = 0.0f;
	float pdfPre = 1;

	for (int k = 1; k < order - 1; k++)
	{
		float pdfTemp = 1;// pdfPre;
		for (int j = 0; j < k; j++)
		{
			pdfTemp *= path.pdf[j];
		}
		for (int j = k + 1; j < order; j++)
		{
			pdfTemp *= path.invPdf[j];
		}
		pdfSum += pdfTemp;
	}
	return pdfSum;

}

/*our bidirectional estimator for the path integral*/
Spectrum evalBDPT(BSDFSamplingRecord &bRec,
	int order, EMeasure measure) const
{
	/* Stop if this component was not requested */
	if (measure != ESolidAngle ||
		Frame::cosTheta(bRec.wi) <= 0 ||
		Frame::cosTheta(bRec.wo) <= 0 ||
		((bRec.component != -1 && bRec.component != 0) ||
		!(bRec.typeMask & EGlossyReflection)))
		return Spectrum(0.0f);

	Vector w0 = bRec.wi;
	Vector woN = bRec.wo;

	Spectrum result(0.0f);

	PathSample pathForward;
	PathSample pathBack;

	//sample the two subpaths
	samplePath(w0, bRec.sampler,order, pathForward, false);
	samplePath(woN, bRec.sampler, order, pathBack, true);


	float outLamda = getInLambda(woN);
	float inLamda = getInLambda(w0);

	PathSimple path;
	Vector wi = bRec.wi;
	Vector wo = bRec.wo;

	Vector tempWi, tempWo;

	//sample count from 1 to order - 1, and the last one is connection
	for (int k = 0; k < order; k++)
	{
		Spectrum resultPathk(1.0);
		float pdfPathK = 1.0F;
		if (k != 0)
		{
			const Vertex &v = pathForward.vList[k - 1];
			if (v.pdf == 0.0) break;
			resultPathk *= pathForward.vList[k - 1].weightAcc;
			pdfPathK = pathForward.vList[k - 1].pdfAcc;
		}

		if (resultPathk.getLuminance() < THRESHOLD || pdfPathK < THRESHOLD)
			break;

		for (int i = 0; i < order - k; i++)
		{
			if (i != 0)
			{
				const Vertex &vt = pathBack.vList[i - 1];
				if (vt.pdf == 0.0) break;
			}

			int currentOrder = k + i + 1;

			//k samples for forward path and j samples from back paths
			if (k == 0 && i == 0)
			{
				result += evalBounce(wi, wo, true);
				continue;
			}
			if (k + i > order - 1)
				continue;

			path.count = 0; //reset the path.
			float pdfForward = 1.0;
			float pdfBackward = 1.0f;
			Spectrum resultPath = resultPathk;
			float pdfPath = pdfPathK;
			pdfForward *= pdfPathK;

			if (k>0)
				pdfBackward *= pathForward.vList[k - 1].invPdfAcc;

			if (i > 0)
			{
				resultPath *= pathBack.vList[i - 1].weightAcc;
				pdfPath *= pathBack.vList[i - 1].invPdfAcc;

				pdfForward *= pathBack.vList[i - 1].pdfAcc;
				pdfBackward *= pathBack.vList[i - 1].invPdfAcc;
			}

			if (resultPath.getLuminance() < THRESHOLD || pdfPath < THRESHOLD)
				break;

			for (int j = 0; j < k; j++)
			{
				const Vertex &v = pathForward.vList[j];
				path.add(v.pdf, v.invPdf);
			}

			if (i == 0)
			{
				//connect the kth to wo
				const Vertex &v = pathForward.vList[k - 1];
				tempWi = -v.wo;
				tempWo = woN;
				float inLamdaTemp = abs(v.outLamda) - 1;
#if OPTIMIZE_PERFORMANCE
				resultPath *= evalBounceLast(tempWi, tempWo, inLamdaTemp, outLamda, false);
#else
				resultPath *= evalBounceLast(tempWi, tempWo);
#endif
				if (pdfBackward > THRESHOLD)
				{
					float pdf = 1;
					float invPdf = pdfWi(woN, -v.wo, outLamda);

					pdfForward *= pdf;
					pdfBackward *= invPdf;
					path.add(pdf, invPdf);
				}
				else
				{
					path.add(1, 0.0f);
				}

			}

			if (k != 0 && i != 0) //th
			{
				//we should connect two of them
				const Vertex &v_s = pathForward.vList[k - 1];
				const Vertex &v_t = pathBack.vList[i - 1];
				Vector tempWi = -v_s.wo;
				Vector tempWo = -v_t.wo;

				float inLamdaTemp = abs(v_s.outLamda) - 1;
				float outLamdaTemp = abs(v_t.outLamda) - 1;

				resultPath *= evalBounceSampleMiddle(tempWi, tempWo, inLamdaTemp, outLamdaTemp);
				float pdf = 0;
				if (pdfForward > THRESHOLD)
				{
					pdf = pdfWi(tempWi, tempWo, inLamdaTemp);
					pdfForward *= pdf;
				}
				float invPdf = 0;
				if (pdfBackward > THRESHOLD)
				{
					tempWo = -v_s.wo;
					tempWi = -v_t.wo;
					invPdf = pdfWi(tempWi, tempWo, outLamdaTemp);
					pdfBackward *= invPdf;
				}
				path.add(pdf, invPdf);

			}

			if (k == 0) //this is the first one
			{
				const Vertex &v = pathBack.vList[i - 1];
				Vector tempWi = w0;
				Vector tempWo = -v.wo;
				float outLamdaTemp = abs(v.outLamda) - 1;

				resultPath *= evalBounceSampleMiddle(tempWi, tempWo, inLamda, outLamdaTemp);
				float invPdf = 1;
				pdfBackward *= invPdf;

				float pdf = 0;
				if (pdfForward > THRESHOLD)
				{
					pdf = pdfWi(tempWi, tempWo, inLamda);
					pdfForward *= pdf;
				}
				path.add(pdf, invPdf);

			}

			for (int j = i - 1; j >= 0; j--)
			{
				const Vertex &v = pathBack.vList[j];
				path.add(v.pdf, v.invPdf);
			}

			if (resultPath.getLuminance() < THRESHOLD || pdfPath < THRESHOLD)
				continue;

			float pdfSum = pdfForward + pdfBackward + computeWeightSum(path, currentOrder);

			result += pdfSum == 0.0 ? Spectrum(0.0f) : resultPath * pdfPath / pdfSum;
		}

	}
	return result;

}



	Spectrum Fresnel(BSDFSamplingRecord &bRec) const
	{
#if FRESNEL
		Vector H = normalize(bRec.wo + bRec.wi);
		const Spectrum F = fresnelConductorExact(dot(bRec.wi, H), m_eta, m_k) * m_specularReflectance->eval(bRec.its);;
		return F;
#else
		return Spectrum(1.0);
#endif
	}

	float computeG1pdf(const Vector3 &wi) const
	{
		float lambda = getLambda(wi);
		float G11 = 1.0f / (1.0f + lambda);

		if (!IsFiniteNumber(G11))
			return 0.0f;
		return G11;
	}

	float evalBouncePdf(const BSDFSamplingRecord &bRec, bool outShadow, bool inShadow) const
	{
		/* Calculate the reflection half-vector */
		Vector H = normalize(bRec.wo + bRec.wi);
		Float D = m_distr->eval(H);
		float G = std::min(1.0f, computeG1pdf(bRec.wi));

		Float model = D * G / (4 * bRec.wi.z);
		return model;
	}

	Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const {
		if (measure != ESolidAngle ||
			Frame::cosTheta(bRec.wi) <= 0 ||
			Frame::cosTheta(bRec.wo) <= 0 ||
			((bRec.component != -1 && bRec.component != 0) ||
			!(bRec.typeMask & EGlossyReflection)))
			return 0.0f;

		//follow how Eric Heitz works: single scattering pdf + diffuse
		float pdf = evalBouncePdf(bRec, true, true) + Frame::cosTheta(bRec.wo);
		return pdf;

	}

	Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &sample) const {
		float pdf;
		return this->sample(bRec, pdf, sample);
	}

	Spectrum sample(BSDFSamplingRecord &bRec, Float &pdf, const Point2 &sample) const {
		if (Frame::cosTheta(bRec.wi) < 0 ||
			((bRec.component != -1 && bRec.component != 0) ||
			!(bRec.typeMask & EGlossyReflection)))
			return Spectrum(0.0f);

		Spectrum weight(1.0f);
		pdf = 1.0f;
		int bounce = 0;
		Vector w0 = bRec.wi;
	
		while (true)
		{
			float currentPdf;
			bRec.wo = sampleWi(bRec.wi, bRec.sampler->next2D(), currentPdf);
			pdf *= currentPdf;
			if (pdf == 0)
				return Spectrum(0.0f);

			bounce++;
			if (weight.isZero()) return Spectrum(0.0f);

			if (Frame::cosTheta(bRec.wo) <= 0)
			{
				//the ray can not leave the surface
				if (bounce == m_order)
				{
					weight = Spectrum(0.0f);
					bRec.wo = Vector3(0, 0, 1);
					break;
				}

				//the ray starts the next bounce
				Spectrum F = Fresnel(bRec);
				weight *= F;

				bRec.wi = -bRec.wo;
			}
			else
			{
				//the ray leave the surface, since the maximum bounce has reached
				if (bounce == m_order)
				{
					float G1 = computeG1(bRec.wo);
					weight *= G1;
					Spectrum F = Fresnel(bRec);
					weight *= F;
					break;
				}
				//the ray continues the tracing with G1 as the probablity
				float G1 = computeG1(bRec.wo);
				Spectrum F = Fresnel(bRec);
				weight *= F;

				float rand = bRec.sampler->next1D();
				if (rand < G1)
				{
					pdf *= G1;
					break;
				}
				else
				{
					pdf *= (1 - G1);
					bRec.wi = -bRec.wo;
				}
			}
		}	
		bRec.wi = w0;
		if (pdf == 0)
			return Spectrum(0.0f);

		bRec.eta = 1.0f;
		bRec.sampledComponent = 0;
		bRec.sampledType = EGlossyReflection;
		pdf = this->pdf(bRec, ESolidAngle);

		return weight;
	}

	void addChild(const std::string &name, ConfigurableObject *child) {
		SLog(EDebug, "%s", name);

		if (child->getClass()->derivesFrom(MTS_CLASS(Texture))) {
			if (name == "alpha")
				m_alphaU = m_alphaV = static_cast<Texture *>(child);
			else if (name == "alphaU")
				m_alphaU = static_cast<Texture *>(child);
			else if (name == "alphaV")
				m_alphaV = static_cast<Texture *>(child);
			else if (name == "specularReflectance")
				m_specularReflectance = static_cast<Texture *>(child);
			else 
				BSDF::addChild(name, child);
		}
		else {
			BSDF::addChild(name, child);
		}
	}

	Float getRoughness(const Intersection &its, int component) const {
		return 0.5f * (m_alphaU->eval(its).average()
			+ m_alphaV->eval(its).average());
	}

	std::string toString() const {
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
	Float getPhongExponent(const Intersection &its) const
	{
		MicrofacetDistribution distr(
			MicrofacetDistribution::EPhong,
			m_alphaU->eval(its).average(),
			m_alphaV->eval(its).average(),
			m_sampleVisible
			);
		return distr.getExponent();
	}
	Shader *createShader(Renderer *renderer) const;

	MTS_DECLARE_CLASS()
private:
	MicrofacetDistribution::EType m_type;
	ref<Texture> m_specularReflectance;
	ref<Texture> m_alphaU, m_alphaV;
	bool m_sampleVisible;
	Spectrum m_eta, m_k;

	float m_alpha_x;
	float m_alpha_y;
	MicrofacetDistribution *m_distr;

	//for mbndf
	int m_order; // the maximum bounces of the multiple scattering
	bool m_bdpt; // using the bidirectional estimator
	bool m_hCorrelated; // using the height correlated model
	int m_rrDepth; // the depth of the Russian Roulette


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
	RoughConductorShader(Renderer *renderer, const Texture *specularReflectance,
		const Texture *alphaU, const Texture *alphaV, const Spectrum &eta,
		const Spectrum &k) : Shader(renderer, EBSDFShader),
		m_specularReflectance(specularReflectance), m_alphaU(alphaU), m_alphaV(alphaV){
		m_specularReflectanceShader = renderer->registerShaderForResource(m_specularReflectance.get());
		m_alphaUShader = renderer->registerShaderForResource(m_alphaU.get());
		m_alphaVShader = renderer->registerShaderForResource(m_alphaV.get());

		/* Compute the reflectance at perpendicular incidence */
		m_R0 = fresnelConductorExact(1.0f, eta, k);
	}

	bool isComplete() const {
		return m_specularReflectanceShader.get() != NULL &&
			m_alphaUShader.get() != NULL &&
			m_alphaVShader.get() != NULL;
	}

	void putDependencies(std::vector<Shader *> &deps) {
		deps.push_back(m_specularReflectanceShader.get());
		deps.push_back(m_alphaUShader.get());
		deps.push_back(m_alphaVShader.get());
	}

	void cleanup(Renderer *renderer) {
		renderer->unregisterShaderForResource(m_specularReflectance.get());
		renderer->unregisterShaderForResource(m_alphaU.get());
		renderer->unregisterShaderForResource(m_alphaV.get());
	}

	void resolve(const GPUProgram *program, const std::string &evalName, std::vector<int> &parameterIDs) const {
		parameterIDs.push_back(program->getParameterID(evalName + "_R0", false));
	}

	void bind(GPUProgram *program, const std::vector<int> &parameterIDs, int &textureUnitOffset) const {
		program->setParameter(parameterIDs[0], m_R0);
	}

	void generateCode(std::ostringstream &oss,
		const std::string &evalName,
		const std::vector<std::string> &depNames) const {
		oss << "uniform vec3 " << evalName << "_R0;" << endl
			<< endl
			<< "float " << evalName << "_D(vec3 m, float alphaU, float alphaV) {" << endl
			<< "    float ct = cosTheta(m), ds = 1-ct*ct;" << endl
			<< "    if (ds <= 0.0)" << endl
			<< "        return 0.0f;" << endl
			<< "    alphaU = 2 / (alphaU * alphaU) - 2;" << endl
			<< "    alphaV = 2 / (alphaV * alphaV) - 2;" << endl
			<< "    float exponent = (alphaU*m.x*m.x + alphaV*m.y*m.y)/ds;" << endl
			<< "    return sqrt((alphaU+2) * (alphaV+2)) * 0.15915 * pow(ct, exponent);" << endl
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
			<< "   float alphaU = max(0.2, " << depNames[1] << "(uv).r);" << endl
			<< "   float alphaV = max(0.2, " << depNames[2] << "(uv).r);" << endl
			<< "   float D = " << evalName << "_D(H, alphaU, alphaV)" << ";" << endl
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
	ref<Shader> m_specularReflectanceShader;
	ref<Shader> m_alphaUShader;
	ref<Shader> m_alphaVShader;
	Spectrum m_R0;
};

Shader *RoughConductor::createShader(Renderer *renderer) const {
	return new RoughConductorShader(renderer,
		m_specularReflectance.get(), m_alphaU.get(), m_alphaV.get(), m_eta, m_k);
}

MTS_IMPLEMENT_CLASS(RoughConductorShader, false, Shader)
MTS_IMPLEMENT_CLASS_S(RoughConductor, false, BSDF)
MTS_EXPORT_PLUGIN(RoughConductor, "Rough conductor BRDF");
MTS_NAMESPACE_END
