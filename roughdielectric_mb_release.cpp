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

#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/sampler.h>
#include <mitsuba/hw/basicshader.h>
#include "microfacet.h"
#include "ior.h"

#include <mitsuba/core/fresolver.h>

MTS_NAMESPACE_BEGIN

#define PI_DIVIDE_180 0.0174532922
#define THRESHOLD 1e-4
#define INV_2_SQRT_M_PI	0.28209479177387814347f /* 0.5/sqrt(pi) */
#define MAX_VERTEX 10

//optimized code
#define G_REUSE 1


/*!\plugin{roughdielectric}{Rough dielectric material}
* \order{5}
* \icon{bsdf_roughdielectric}
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
*     \parameter{intIOR}{\Float\Or\String}{Interior index of refraction specified
*         numerically or using a known material name. \default{\texttt{bk7} / 1.5046}}
*     \parameter{extIOR}{\Float\Or\String}{Exterior index of refraction specified
*         numerically or using a known material name. \default{\texttt{air} / 1.000277}}
*     \parameter{sampleVisible}{\Boolean}{
*         Enables a sampling technique proposed by Heitz and D'Eon~\cite{Heitz1014Importance},
*         which focuses computation on the visible parts of the microfacet normal
*         distribution, considerably reducing variance in some cases.
*         \default{\code{true}, i.e. use visible normal sampling}
*     }
*     \parameter{specular\showbreak Reflectance,\newline
*         specular\showbreak Transmittance}{\Spectrum\Or\Texture}{Optional
*         factor that can be used to modulate the specular reflection/transmission component. Note
*         that for physical realism, this parameter should never be touched. \default{1.0}}
* }\vspace{4mm}
*
* This plugin implements a realistic microfacet scattering model for rendering
* rough interfaces between dielectric materials, such as a transition from air to
* ground glass. Microfacet theory describes rough surfaces as an arrangement of
* unresolved and ideally specular facets, whose normal directions are given by
* a specially chosen \emph{microfacet distribution}. By accounting for shadowing
* and masking effects between these facets, it is possible to reproduce the important
* off-specular reflections peaks observed in real-world measurements of such
* materials.
* \renderings{
*     \rendering{Anti-glare glass (Beckmann, $\alpha=0.02$)}
*     	   {bsdf_roughdielectric_beckmann_0_0_2.jpg}
*     \rendering{Rough glass (Beckmann, $\alpha=0.1$)}
*     	   {bsdf_roughdielectric_beckmann_0_1.jpg}
* }
*
* This plugin is essentially the ``roughened'' equivalent of the (smooth) plugin
* \pluginref{dielectric}. For very low values of $\alpha$, the two will
* be identical, though scenes using this plugin will take longer to render
* due to the additional computational burden of tracking surface roughness.
*
* The implementation is based on the paper ``Microfacet Models
* for Refraction through Rough Surfaces'' by Walter et al.
* \cite{Walter07Microfacet}. It supports three different types of microfacet
* distributions and has a texturable roughness parameter. Exterior and
* interior IOR values can be specified independently, where ``exterior''
* refers to the side that contains the surface normal. Similar to the
* \pluginref{dielectric} plugin, IOR values can either be specified
* numerically, or based on a list of known materials (see
* \tblref{dielectric-iors} for an overview). When no parameters are given,
* the plugin activates the default settings, which describe a borosilicate
* glass BK7/air interface with a light amount of roughness modeled using a
* Beckmann distribution.
*
* To get an intuition about the effect of the surface roughness parameter
* $\alpha$, consider the following approximate classification: a value of
* $\alpha=0.001-0.01$ corresponds to a material with slight imperfections
* on an otherwise smooth surface finish, $\alpha=0.1$ is relatively rough,
* and $\alpha=0.3-0.7$ is \emph{extremely} rough (e.g. an etched or ground
* finish). Values significantly above that are probably not too realistic.
*
* Please note that when using this plugin, it is crucial that the scene contains
* meaningful and mutually compatible index of refraction changes---see
* \figref{glass-explanation} for an example of what this entails. Also, note that
* the importance sampling implementation of this model is close, but
* not always a perfect a perfect match to the underlying scattering distribution,
* particularly for high roughness values and when the \texttt{ggx}
* microfacet distribution is used. Hence, such renderings may
* converge slowly.
*
* \subsubsection*{Technical details}
* All microfacet distributions allow the specification of two distinct
* roughness values along the tangent and bitangent directions. This can be
* used to provide a material with a ``brushed'' appearance. The alignment
* of the anisotropy will follow the UV parameterization of the underlying
* mesh. This means that such an anisotropic material cannot be applied to
* triangle meshes that are missing texture coordinates.
*
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
* When rendering with the Phong microfacet distribution, a conversion is
* used to turn the specified Beckmann-equivalent $\alpha$ roughness value
* into the exponents of the distribution. This is done in a way, such that
* the different distributions all produce a similar appearance for the
* same value of $\alpha$.
*
* \renderings{
*     \rendering{Ground glass (GGX, $\alpha$=0.304,
*     	   \lstref{roughdielectric-roughglass})}{bsdf_roughdielectric_ggx_0_304.jpg}
*     \rendering{Textured roughness (\lstref{roughdielectric-textured})}
*         {bsdf_roughdielectric_textured.jpg}
* }
*
* \begin{xml}[caption=A material definition for ground glass, label=lst:roughdielectric-roughglass]
* <bsdf type="roughdielectric">
*     <string name="distribution" value="ggx"/>
*     <float name="alpha" value="0.304"/>
*     <string name="intIOR" value="bk7"/>
*     <string name="extIOR" value="air"/>
* </bsdf>
* \end{xml}
*
* \begin{xml}[caption=A texture can be attached to the roughness parameter, label=lst:roughdielectric-textured]
* <bsdf type="roughdielectric">
*     <string name="distribution" value="beckmann"/>
*     <float name="intIOR" value="1.5046"/>
*     <float name="extIOR" value="1.0"/>
*
*     <texture name="alpha" type="bitmap">
*         <string name="filename" value="roughness.exr"/>
*     </texture>
* </bsdf>
* \end{xml}
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
	Vertex(const Vector &_wi, const Vector &_wo, float _pdf, const float &_weight,
		bool _outside, bool _reflect, const float& _specAcc, float _pdfAcc,
		float _inLamda, float _outLamda, float _invPdf, float _invPdfAcc)
	{
		wi = _wi;
		wo = _wo;
		pdf = _pdf;
		weight = _weight;
		outside = _outside;
		reflect = _reflect;
		weightAcc = _specAcc;
		pdfAcc = _pdfAcc;
		inLamda = _inLamda;
		outLamda = _outLamda;
		invPdf = _invPdf;
		invPdfAcc = _invPdfAcc;
	}
	Vector wi;
	Vector wo;
	float weight;
	float weightAcc;

	float pdf;
	float pdfAcc;
	float invPdf;
	float invPdfAcc;

	bool outside; // outside for wi.
	bool reflect;
	float inLamda;
	float outLamda;
};

struct PathSample{
	PathSample(){ count = 0; }

	void add(const Vector &_wi, const Vector &_wo, float _pdf,
		const float &_weight, bool _outside, bool _type,
		const float& _specAcc, float _pdfAcc,
		float inLamda, float outLamda, float invPdf, float _invPdfAcc)
	{
		vList[count] = Vertex(_wi, _wo, _pdf, _weight, _outside, _type, _specAcc, _pdfAcc, inLamda, outLamda, invPdf, _invPdfAcc);
		count++;
	}

	void add(float _pdf){
		vList[count] = Vertex(_pdf);
		count++;
	}
	Vertex vList[MAX_VERTEX];
	int count;
};


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

static inline float generateRandomNumber()
{
	const float U = ((float)rand()) / (float)RAND_MAX;
	return U;
}


static inline bool IsFiniteNumber(float x)
{
	return (x <= std::numeric_limits<Float>::max() && x >= -std::numeric_limits<Float>::max());
}

#define vec3 Vector
#define vec2 Vector
vec2 sampleP22_11(const float theta_i, const float U, const float U_2, const float alpha_x, const float alpha_y)
{
	vec2 slope;

	if (theta_i < 0.0001f)
	{
		const float r = sqrtf(U / (1.0f - U));
		const float phi = 6.28318530718f * U_2;
		slope.x = r * cosf(phi);
		slope.y = r * sinf(phi);
		return slope;
	}

	// constant
	const float sin_theta_i = sinf(theta_i);
	const float cos_theta_i = cosf(theta_i);
	const float tan_theta_i = sin_theta_i / cos_theta_i;

	// slope associated to theta_i
	const float slope_i = cos_theta_i / sin_theta_i;

	// projected area
	const float projectedarea = 0.5f * (cos_theta_i + 1.0f);
	if (projectedarea < 0.0001f || projectedarea != projectedarea)
		return vec2(0, 0, 0);
	// normalization coefficient
	const float c = 1.0f / projectedarea;

	const float A = 2.0f*U / cos_theta_i / c - 1.0f;
	const float B = tan_theta_i;
	const float tmp = 1.0f / (A*A - 1.0f);

	const float D = sqrtf(std::max(0.0f, B*B*tmp*tmp - (A*A - B*B)*tmp));
	const float slope_x_1 = B*tmp - D;
	const float slope_x_2 = B*tmp + D;
	slope.x = (A < 0.0f || slope_x_2 > 1.0f / tan_theta_i) ? slope_x_1 : slope_x_2;

	float U2;
	float S;
	if (U_2 > 0.5f)
	{
		S = 1.0f;
		U2 = 2.0f*(U_2 - 0.5f);
	}
	else
	{
		S = -1.0f;
		U2 = 2.0f*(0.5f - U_2);
	}
	const float z = (U2*(U2*(U2*0.27385f - 0.73369f) + 0.46341f)) / (U2*(U2*(U2*0.093073f + 0.309420f) - 1.000000f) + 0.597999f);
	slope.y = S * z * sqrtf(1.0f + slope.x*slope.x);

	return slope;
}

Vector sampleD_GGX(const Vector &wi, float alpha_x, float alpha_y)
{
	const float U1 = generateRandomNumber();
	const float U2 = generateRandomNumber();

	// sample D_wi

	// stretch to match configuration with alpha=1.0	
	const Vector wi_11 = normalize(Vector(alpha_x * wi.x, alpha_y * wi.y, wi.z));

	// sample visible slope with alpha=1.0
	Vector slope_11 = sampleP22_11(acosf(wi_11.z), U1, U2, alpha_x, alpha_y);

	// align with view direction
	const float phi = atan2(wi_11.y, wi_11.x);
	Vector slope(cos(phi)*slope_11.x - sinf(phi)*slope_11.y, sinf(phi)*slope_11.x + cos(phi)*slope_11.y, 0.0f);

	// stretch back
	slope.x *= alpha_x;
	slope.y *= alpha_y;

	// compute normal
	Vector3 wm;
	// if numerical instability
	if ((slope.x != slope.x) || !IsFiniteNumber(slope.x))
	{
		if (wi.z > 0) wm = Vector3(0.0f, 0.0f, 1.0f);
		else wm = normalize(Vector3(wi.x, wi.y, 0.0f));
	}
	else
		wm = normalize(Vector3(-slope.x, -slope.y, 1.0f));

	return wm;
}


class RoughDielectric : public BSDF {
public:
	RoughDielectric(const Properties &props) : BSDF(props) {
		m_specularReflectance = new ConstantSpectrumTexture(
			props.getSpectrum("specularReflectance", Spectrum(1.0f)));
		m_specularTransmittance = new ConstantSpectrumTexture(
			props.getSpectrum("specularTransmittance", Spectrum(1.0f)));

		/* Specifies the internal index of refraction at the interface */
		Float intIOR = lookupIOR(props, "intIOR", "bk7");

		/* Specifies the external index of refraction at the interface */
		Float extIOR = lookupIOR(props, "extIOR", "air");

		if (intIOR < 0 || extIOR < 0 || intIOR == extIOR)
			Log(EError, "The interior and exterior indices of "
			"refraction must be positive and differ!");

		m_eta = intIOR / extIOR;
		m_invEta = 1 / m_eta;
		MicrofacetDistribution distr(props);
		m_type = distr.getType();
		m_sampleVisible = distr.getSampleVisible();

		m_alphaU = new ConstantFloatTexture(distr.getAlphaU());
		if (distr.getAlphaU() == distr.getAlphaV())
			m_alphaV = m_alphaU;
		else
			m_alphaV = new ConstantFloatTexture(distr.getAlphaV());

		m_order = props.getInteger("order", 1);
		m_rrDepth = props.getInteger("rrDepth", m_order);
		m_bdpt = props.getBoolean("BDPT", false);		
		m_hCorrelated = props.getBoolean("hCorrelated", false);

		Intersection its;
		m_alpha_x = m_alphaU->eval(its).average();
		m_alpha_y = m_alphaV->eval(its).average();
		m_distr = new MicrofacetDistribution
			(
			m_type,
			m_alpha_x,
			m_alpha_y,
			true
			);
	}

	RoughDielectric(Stream *stream, InstanceManager *manager)
		: BSDF(stream, manager) {
		m_type = (MicrofacetDistribution::EType) stream->readUInt();
		m_sampleVisible = stream->readBool();
		m_alphaU = static_cast<Texture *>(manager->getInstance(stream));
		m_alphaV = static_cast<Texture *>(manager->getInstance(stream));
		m_specularReflectance = static_cast<Texture *>(manager->getInstance(stream));
		m_specularTransmittance = static_cast<Texture *>(manager->getInstance(stream));
		m_eta = stream->readFloat();
		m_invEta = 1 / m_eta;

		configure();
	}

	void serialize(Stream *stream, InstanceManager *manager) const {
		BSDF::serialize(stream, manager);

		stream->writeUInt((uint32_t)m_type);
		stream->writeBool(m_sampleVisible);
		manager->serialize(stream, m_alphaU.get());
		manager->serialize(stream, m_alphaV.get());
		manager->serialize(stream, m_specularReflectance.get());
		manager->serialize(stream, m_specularTransmittance.get());
		stream->writeFloat(m_eta);
	}

	void configure() {
		unsigned int extraFlags = 0;
		if (m_alphaU != m_alphaV)
			extraFlags |= EAnisotropic;

		if (!m_alphaU->isConstant() || !m_alphaV->isConstant())
			extraFlags |= ESpatiallyVarying;

		m_components.clear();
		m_components.push_back(EGlossyReflection | EFrontSide
			| EBackSide | EUsesSampler | extraFlags
			| (m_specularReflectance->isConstant() ? 0 : ESpatiallyVarying));
		m_components.push_back(EGlossyTransmission | EFrontSide
			| EBackSide | EUsesSampler | ENonSymmetric | extraFlags
			| (m_specularTransmittance->isConstant() ? 0 : ESpatiallyVarying));

		/* Verify the input parameters and fix them if necessary */
		m_specularReflectance = ensureEnergyConservation(
			m_specularReflectance, "specularReflectance", 1.0f);
		m_specularTransmittance = ensureEnergyConservation(
			m_specularTransmittance, "specularTransmittance", 1.0f);

		m_usesRayDifferentials =
			m_alphaU->usesRayDifferentials() ||
			m_alphaV->usesRayDifferentials() ||
			m_specularReflectance->usesRayDifferentials() ||
			m_specularTransmittance->usesRayDifferentials();

		BSDF::configure();
	}

	static inline double  abgam(double x)
	{
		double  gam[10],
			temp;

		gam[0] = 1. / 12.;
		gam[1] = 1. / 30.;
		gam[2] = 53. / 210.;
		gam[3] = 195. / 371.;
		gam[4] = 22999. / 22737.;
		gam[5] = 29944523. / 19733142.;
		gam[6] = 109535241009. / 48264275462.;
		temp = 0.5*log(2 * M_PI) - x + (x - 0.5)*log(x)
			+ gam[0] / (x + gam[1] / (x + gam[2] / (x + gam[3] / (x + gam[4] /
			(x + gam[5] / (x + gam[6] / x))))));

		return temp;
	}

	static inline double  gamma(double x)
	{
		double  result;
		result = exp(abgam(x + 5)) / (x*(x + 1)*(x + 2)*(x + 3)*(x + 4));
		return result;
	}

	static inline double  beta(double m, double n)
	{
		return (gamma(m)*gamma(n) / gamma(m + n));
	}

	float getLambda(const Vector3 &wo, float alpha_x, float alpha_y) const
	{
		if (m_type == MicrofacetDistribution::EBeckmann)
		{
			RayInfoBeckmann ray;

			ray.updateDirection(wo, alpha_x, alpha_y);
			return ray.Lambda;
		}
		else
		{
		RayInfoGGX ray;
		ray.updateDirection(wo, alpha_x, alpha_y);

		return ray.Lambda;
	}
	}

	float getInLambda(const Vector3 &wi, float alpha_x, float alpha_y) const
	{
		if (m_type == MicrofacetDistribution::EBeckmann)
		{
			RayInfoBeckmann ray;
			ray.updateDirection(-wi, alpha_x, alpha_y);
			return abs(ray.Lambda) - 1;
		}
		{
		RayInfoGGX ray;
		ray.updateDirection(-wi, alpha_x, alpha_y);
		return abs(ray.Lambda) - 1;
	}

	}

	void getLambda(const Vector3 &wi, const Vector3 &wo, float alpha_x, float alpha_y, float &inLambda, float &outLambda) const
	{
		if (m_type == MicrofacetDistribution::EBeckmann)
		{
			RayInfoBeckmann ray;
			ray.updateDirection(-wi, alpha_x, alpha_y);
			RayInfoBeckmann ray_shadowing;
			ray_shadowing.updateDirection(wo, alpha_x, alpha_y);
			inLambda = (abs(-ray.Lambda) - 1.0f);
			outLambda = ray_shadowing.Lambda;
		}
		else
		{
			RayInfoGGX ray;
			ray.updateDirection(-wi, alpha_x, alpha_y);
			RayInfoGGX ray_shadowing;
			ray_shadowing.updateDirection(wo, alpha_x, alpha_y);
			inLambda = (abs(-ray.Lambda) - 1.0f);
			outLambda = ray_shadowing.Lambda;
		}

	}

	float computeG2_R_last_cor(const Vector3 &wi, const Vector &wo) const
	{
		float inLambda, outLambda;
		getLambda(wi, wo, m_alpha_x, m_alpha_y, inLambda, outLambda);
		float G = (float)beta(1.0f + inLambda, 1.0f + outLambda);
		return G;

	}


	float computeG1(const Vector3 &wi) const
	{
		float lambda = getInLambda(wi, m_alpha_x, m_alpha_y);
		float G11 = 1.0f / (1.0f + lambda);
		return G11;

	}

	//this is for reflect in G, out 1-G
	float computeG2_R_middle_cor(const Vector3 &wi, const Vector &wo) const
	{
		float inLambda, outLambda;
		getLambda(wi, wo, m_alpha_x, m_alpha_y, inLambda, outLambda);

		const float Gtemp = 1.0f / (1.0f + inLambda + outLambda);
		const float Gtemp2 = 1.0f / (1.0f + inLambda);
		float  G = std::max(0.0f, Gtemp2 - Gtemp);
		return G;

	}

	float computeG2_last_un(const Vector3 &wi, const Vector &wo) const
	{
		float inLambda, outLambda;
		getLambda(wi, wo, m_alpha_x, m_alpha_y, inLambda, outLambda);
		float G11 = abs(1.0f / (1.0f + inLambda));
		float G12 = 1.0f / (1.0f + outLambda);
		float G = G11*G12;
		return G;

	}

	float computeG2_R_middle_cor_opt(const float inLamda, const float outLamda) const
	{
		const float Gtemp = 1.0f / (1.0 + inLamda + outLamda);
		const float Gtemp2 = 1.0f / (1.0 + inLamda);
		float  G = Gtemp2 - Gtemp;
		return G;
	}


	float computeG2_T_middle_cor_opt(const float inLamda, const float outLamda) const
	{
		float temp = 1.0f + inLamda;
		const float Gtemp = (float)beta(temp, 1.0f + outLamda);
		const float Gtemp2 = 1.0f / temp;
		float G = std::min(1.0f, std::max(0.0f, Gtemp2 - Gtemp));
		return G;
	}
	

	float computeG2_T_last_cor(const Vector3 &wi, const Vector &wo) const
	{
		float inLambda, outLambda;
		getLambda(wi, wo, m_alpha_x, m_alpha_y, inLambda, outLambda);

		float temp = 1.0f + inLambda;
		const float G = (float)beta(temp, 1.0f + outLambda);
		return G;
	}
	float computeG2_T_middle_cor(const Vector3 &wi, const Vector &wo) const
	{

		float inLambda, outLambda;
		getLambda(wi, wo, m_alpha_x, m_alpha_y, inLambda, outLambda);

		float temp = 1.0f + inLambda;
		const float Gtemp = (float)beta(temp, 1.0f + outLambda);
		const float Gtemp2 = 1.0f / temp;
		float G = Gtemp2 - Gtemp;// 
		return G;

	}

	//for the middle bounce
	float computeG2_R_middle(BSDFSamplingRecord &bRec, bool outside, float inLamda, float outLamda)const
	{
		float G;
		if (outside ? bRec.wo.z < 0 : bRec.wo.z > 0)
		{
#if G_REUSE
			G = 1.0f / (1.0f + inLamda);
#else
			G = computeG1(outside ? bRec.wi : -bRec.wi);
#endif
		}
		else
		{
			if (m_hCorrelated)
			{

#if G_REUSE
				G = computeG2_R_middle_cor_opt(inLamda, outLamda);
#else
				G = computeG2_R_middle_cor(outside ? bRec.wi : -bRec.wi, outside ? bRec.wo : -bRec.wo);
#endif
			}
			else
			{
#if G_REUSE
				float G11 = 1.0f / (1.0f + inLamda);
				float G12 = 1 - 1.0f / (1.0f + outLamda);
#else
				float G11 = computeG1(outside ? bRec.wi : -bRec.wi);
				float G12 = 1 - std::min(1.0f, computeG1(outside ? bRec.wo : -bRec.wo));
#endif

				G = G11*G12;
			}
		}

		if (!IsFiniteNumber(G))
			return 0;
		return G;

	}

	Spectrum eval(BSDFSamplingRecord &bRec, EMeasure measure) const {
		if (measure != ESolidAngle || Frame::cosTheta(bRec.wi) == 0)
			return Spectrum(0.0f);
		return  m_bdpt ? evalBDPT(bRec, m_order, measure) : evalPT(bRec, m_order, measure);
	}


	//we compute the full pdf here for bidirectional case
	inline Vector sampleVNDF_Full(BSDFSamplingRecord &bRec,
		float &pdf, float &simplePdf, bool outside, float inLamda, float alpha_x, float alpha_y) const
	{
		Normal m;


		m =  m_type == MicrofacetDistribution::EBeckmann ?
			m_distr->sample(outside ? bRec.wi : -bRec.wi, bRec.sampler->next2D())
			:sampleD_GGX(outside ? bRec.wi : -bRec.wi, alpha_x, alpha_y);

		//m_distr->sample(outside ? bRec.wi : -bRec.wi, bRec.sampler->next2D());// , pdf);
		if (dot(m, bRec.wi) == 0.0)
		{
			pdf = 0.0;
			return m;
		}

		float D = m_distr->eval(m);

		if (D == 0.0)
		{
			pdf = 0.0;
			return m;
		}

#if G_REUSE
		float G1 = 1.0 / (1 + inLamda); 
#else
		float G1 = computeG1(outside ? bRec.wi : -bRec.wi);// 
#endif

		if (!IsFiniteNumber(G1))
		{
			pdf = 0.0f;
			return m;
		}
		simplePdf = G1;
		pdf = G1 *absDot(bRec.wi, m) * D / std::abs(Frame::cosTheta(bRec.wi));


		Float cosThetaT;
		Float F = fresnelDielectricExt(dot(bRec.wi, m), cosThetaT, m_eta);
		Float dwh_dwo;
		if (bRec.sampler->next1D() < F)
		{
			bRec.wo = reflect(bRec.wi, m);
			bRec.eta = 1.0f;
			bRec.sampledComponent = 0;
			bRec.sampledType = EGlossyReflection;
			pdf *= F;
			dwh_dwo = 1.0f / (4.0f * dot(bRec.wo, m));
		}
		else
		{
			if (cosThetaT == 0)
				pdf = 0.0;

			/* Perfect specular transmission based on the microfacet normal */

			bRec.wo = refract(bRec.wi, m, m_eta, cosThetaT);
			bRec.eta = cosThetaT < 0 ? m_eta : m_invEta;
			bRec.sampledComponent = 1;
			bRec.sampledType = EGlossyTransmission;
			pdf *= (1 - F);

			/* Calculate the transmission half-vector */
			Float eta = outside > 0 ? m_eta : m_invEta;
			Float sqrtDenom = dot(bRec.wi, m) + eta * dot(bRec.wo, m);
			dwh_dwo = (eta*eta * dot(bRec.wo, m)) / (sqrtDenom*sqrtDenom);
		}

		pdf *= std::abs(dwh_dwo);
		return m;
	}

	inline void sampleVNDF(BSDFSamplingRecord &bRec, float &pdf, bool outside, float inLamda, float alpha_x, float alpha_y) const
	{
		Normal m = (m_type == MicrofacetDistribution::EBeckmann) ?
			m_distr->sample(outside ? bRec.wi : -bRec.wi, bRec.sampler->next2D()) 
			: sampleD_GGX(outside ? bRec.wi : -bRec.wi, alpha_x, alpha_y);

		if (m.z <= 1e-3)
		{
			pdf = 0.0;
			return;
		}

#if G_REUSE
		float G1 = 1.0 / (1 + inLamda);
#else
		float G1 = computeG1(outside ? bRec.wi : -bRec.wi);// 
#endif

		if (!IsFiniteNumber(G1))
		{
			pdf = 0.0f;
			return;
		}
		pdf = G1;


		Float cosThetaT;
		Float F = fresnelDielectricExt(dot(bRec.wi, m), cosThetaT, m_eta);

		if (bRec.sampler->next1D() < F)
		{
			bRec.wo = reflect(bRec.wi, m);
			bRec.eta = 1.0f;
			bRec.sampledComponent = 0;
			bRec.sampledType = EGlossyReflection;
		}
		else
		{
			if (cosThetaT == 0)
				pdf = 0.0;

			bRec.wo = refract(bRec.wi, m, m_eta, cosThetaT);
			bRec.eta = cosThetaT < 0 ? m_eta : m_invEta;
			bRec.sampledComponent = 1;
			bRec.sampledType = EGlossyTransmission;

		}
	}


	float evalBounce_R_middle(BSDFSamplingRecord &bRec, bool outside, float inLamda, float outLamda) const
	{
		float result = computeD_F(bRec, outside);
		if (result == 0.0)
			return (0.0f);

		float G = computeG2_R_middle(bRec, outside, inLamda, outLamda);
		result *= G;
		return result;
	}

	float evalBounce_T_middle(BSDFSamplingRecord &bRec, bool outside, float inLamda, float outLamda) const
	{
		float result = computeD_F_T(bRec, outside);
		if (result == 0.0)
			return (0.0f);

		float G = computeG2_T_middle(bRec, outside, inLamda, outLamda);
		result *= G;
		return result;
	}

	//the outside means the side of the wi
	float evalBounce_T_last(BSDFSamplingRecord &bRec, bool outside, float inLamda, float outLamda) const
	{
		float result = computeD_F_T(bRec, outside);
		if (result == 0.0)
			return (0.0f);

#if G_REUSE
		float G = m_hCorrelated ? (float)beta(1.0f + inLamda, 1.0f + outLamda) :
			1.0f / ((1.0f + inLamda)*(1.0f + outLamda));
#else
		float G;
		if (m_hCorrelated)
			G = computeG2_T_last_cor(outside ? bRec.wi : -bRec.wi, (!outside) ? bRec.wo : -bRec.wo);
		else
			G = computeG2_last_un(outside ? bRec.wi : -bRec.wi, (!outside) ? bRec.wo : -bRec.wo);
#endif

		if (!IsFiniteNumber(G))
			return (0.0f);

		return result * G;
	}

	float evalBounce_R_last(BSDFSamplingRecord &bRec, bool outside, float inLamda, float outLamda) const
	{
		float result = computeD_F(bRec, outside);
		if (result == 0.0)
			return (0.0f);

#if G_REUSE
		float G = m_hCorrelated ? 1.0 / (1.0f + inLamda + outLamda) : 1.0f / ((1.0f + inLamda)*(1.0f + outLamda));
#else
		float G;
		if (m_hCorrelated)
		{
			G = computeG2_R_last_cor(outside ? bRec.wi : -bRec.wi, outside ? bRec.wo : -bRec.wo);
		}
		else
		{
			G = computeG2_last_un(outside ? bRec.wi : -bRec.wi, outside ? bRec.wo : -bRec.wo);
		}
#endif

		if (!IsFiniteNumber(G))
			return 0.0f;

		result *= G;
		return result;
	}

	float evalBounce_last(BSDFSamplingRecord &bRec, bool outside,
		float inLamda, float outLamda) const
	{
		float weight;
		if (bRec.sampledType == EGlossyReflection)
		{
			weight = evalBounce_R_last(bRec, outside, inLamda, outLamda);
		}
		else
		{
			weight = evalBounce_T_last(bRec, outside, inLamda, outLamda);
		}
		return weight;
	}

	float computeG2_T_middle(BSDFSamplingRecord &bRec, bool outside, float inLamda, float outLamda) const
	{

		bool currentOutside = !outside;
		float G;
		bool needG = (currentOutside) ? bRec.wo.z > 0 : bRec.wo.z < 0;
		if (!needG)
		{
#if G_REUSE
			G = 1.0f / (1.0f + inLamda);
#else
			G = computeG1(outside ? bRec.wi : -bRec.wi);
#endif
		}
		else
		{
			if (m_hCorrelated)
			{
#if G_REUSE
				G = computeG2_T_middle_cor_opt(inLamda, outLamda);
#else
				G = computeG2_T_middle_cor(outside ? bRec.wi : -bRec.wi, currentOutside ? bRec.wo : -bRec.wo);
#endif
			}
			else
			{
#if G_REUSE
				float G11 = 1.0f / (1.0f + inLamda);
				float G12 = 1 - 1.0f / (1.0f + outLamda);
#else
				float G11 = computeG1(outside ? bRec.wi : -bRec.wi);
				float G12 = 1 - std::min(1.0f, computeG1(currentOutside ? bRec.wo : -bRec.wo));
#endif

				G = G11*G12;
			}
		}
		if (!IsFiniteNumber(G))
			G = 0;
		return G;

	}


	float pdfWi(BSDFSamplingRecord &bRec, bool outside, float inLamda, float alpha_x, float alpha_y) const
	{

		Vector m;
		if (bRec.sampledType == EGlossyReflection)
		{
			/* Calculate the reflection half-vector */
			m = normalize(bRec.wo + bRec.wi);
			/* Ensure that the half-vector points into the
			same hemisphere as the macrosurface normal */
			m *= math::signum(Frame::cosTheta(m));
		}
		else
		{
			Float eta = outside// Frame::cosTheta(bRec.wi) > 0
				? m_eta : m_invEta;
			//FIXME: not sure about this, since the incoming light could come from the down side

			m = normalize(bRec.wi + bRec.wo*eta);
			/* Ensure that the half-vector points into the
			same hemisphere as the macrosurface normal */
			//there might be a problem for more bounces
			m *= math::signum(Frame::cosTheta(m));
		}
		float pdf;
#if G_REUSE
		float G1 = 1.0 / (1 + inLamda);
#else
		float G1 = computeG1(outside ? bRec.wi : -bRec.wi);// 
#endif		
		if (!IsFiniteNumber(G1))
		{
			return 0;
		}

		pdf = G1 *absDot(bRec.wi, m) * m_distr->eval(m) / std::abs(Frame::cosTheta(bRec.wi));

		Float cosThetaT;
		Float F = fresnelDielectricExt(dot(bRec.wi, m), cosThetaT, m_eta);
		Float dwh_dwo;
		if (bRec.sampledType == EGlossyReflection)
		{
			pdf *= F;
			dwh_dwo = 1.0f / (4.0f * dot(bRec.wo, m));
		}
		else
		{
			if (cosThetaT == 0)
				pdf = 0.0;

			/* Perfect specular transmission based on the microfacet normal */
			pdf *= (1 - F);

			/* Calculate the transmission half-vector */
			Float eta = outside > 0 ? m_eta : m_invEta;
			Float sqrtDenom = dot(bRec.wi, m) + eta * dot(bRec.wo, m);
			dwh_dwo = (eta*eta * dot(bRec.wo, m)) / (sqrtDenom*sqrtDenom);
		}

		pdf *= std::abs(dwh_dwo);

		return pdf;
	}

	void samplePath(BSDFSamplingRecord &bRec, int order, PathSample &path,
		bool outside, float alpha_x, float alpha_y)const{

		float pdf = 1.0f;
		float invPdfAcc = 1.0f;
		float pdfAcc = 1.0f;
		float weightAcc(1.0f);
		float eta = 1.0f;
		float inLamda = getInLambda(outside ? bRec.wi : -bRec.wi, alpha_x, alpha_y);

		for (int i = 0; i < order - 1; i++)
		{
			float currentPDF;
			float simplePDF;

			sampleVNDF_Full(bRec, currentPDF, simplePDF, outside, inLamda, alpha_x, alpha_y);
			eta *= bRec.eta;
			pdf *= currentPDF;

			if (pdf > 1e-2 && IsFiniteNumber(currentPDF))
			{
				float weight;
				Vector wo = bRec.wo;
				bool cuOutside = (bRec.sampledType == EGlossyReflection) ? outside : !outside;
				float outLamda = getLambda(cuOutside ? wo : -wo, alpha_x, alpha_y);
				float readyOutLamda = abs(outLamda + 1) - 1;

				weight = evalBounceSample(bRec, outside, inLamda, readyOutLamda) / simplePDF;

				if (!IsFiniteNumber(weight))
				{
					path.add(0.0);
					break;
				}

				weightAcc *= weight;
				if (weightAcc <= 1e-4)
				{
					path.add(0.0);
					break;
				}
				float backPdf = 1;
				if (invPdfAcc > 1e-2)
				{
					bRec.wo = bRec.wi;
					bRec.wi = wo;

					backPdf = i == 0 ? 1 : pdfWi(bRec, cuOutside, readyOutLamda, alpha_x, alpha_y);
					invPdfAcc *= backPdf;
				}

				path.add(bRec.wo, bRec.wi, currentPDF, weight, outside,
					bRec.sampledType == EGlossyReflection, weightAcc, pdf, inLamda, outLamda, backPdf, invPdfAcc);

				outside = cuOutside;
				bRec.wi = -wo;
				inLamda = abs(outLamda) - 1;
			}
			else
			{
				path.add(0.0);
				break;

			}
			if (m_rrDepth > -1 && i + 1 >= m_rrDepth) {
				Float q = std::min(weightAcc * eta * eta, (Float) 0.95f);

				if (bRec.sampler->next1D() > q) {
					path.add(0.0);
					break;
				}
				else {
					weightAcc /= q;
				}
			}
		}
	}


	void samplePath_back(BSDFSamplingRecord &bRec, int order, PathSample &path,
		bool outside, float alpha_x, float alpha_y)const{

		float pdf = 1.0f;
		float invPdfAcc = 1.0f;
		float pdfAcc = 1.0f;
		float weightAcc(1.0f);
		float inLamda = getInLambda(outside ? bRec.wi : -bRec.wi, alpha_x, alpha_y);
		float eta = 1.0f;
		for (int i = 0; i < order - 1; i++)
		{
			float currentPDF;
			float simplePDF;

			sampleVNDF_Full(bRec, currentPDF, simplePDF, outside, inLamda, alpha_x, alpha_y);
			eta *= bRec.eta;
			pdf *= currentPDF;

			if (pdf > 1e-2 && IsFiniteNumber(currentPDF))
			{
				float weight;
				Vector wo = bRec.wo;
				bool cuOutside = (bRec.sampledType == EGlossyReflection) ? outside : !outside;
				float outLamda = getLambda(cuOutside ? wo : -wo, alpha_x, alpha_y);
				float readyOutLamda = abs(outLamda + 1) - 1;
				{
					bRec.wo = bRec.wi;
					bRec.wi = wo;

					if (i == 0) //outshadow
						weight = evalBounce_last(bRec, cuOutside, readyOutLamda, inLamda);
					else
						weight = evalBounce_middle(bRec, cuOutside, readyOutLamda, inLamda, alpha_x, alpha_y);

					weight = weight / currentPDF;
					weightAcc *= weight;
					if (weightAcc <= 1e-4)
					{
						path.add(0.0);
						break;
					}

					float forwardPdf = 1;
					if (pdfAcc > 1e-2)
					{
						forwardPdf = i == 0 ? 1.0 : pdfWi(bRec, cuOutside, readyOutLamda, alpha_x, alpha_y);
						pdfAcc *= forwardPdf;
					}
					path.add(bRec.wo, bRec.wi, forwardPdf, weight, outside,
						bRec.sampledType == EGlossyReflection, weightAcc, pdfAcc, inLamda, outLamda, currentPDF, pdf);
				}

				outside = cuOutside;
				bRec.wi = -wo;
				inLamda = abs(outLamda) - 1;
			}
			else
			{
				path.add(0.0);
				break;

			}
			if (m_rrDepth > -1 && i + 1 >= m_rrDepth) {
				Float q = std::min(weightAcc * eta * eta, (Float) 0.95f);

				if (bRec.sampler->next1D() > q) {
					path.add(0.0);
					break;
				}
				else {
					weightAcc /= q;
				}
			}
		}

	}

	float evalBounce_middle(BSDFSamplingRecord &bRec, bool outside, float inLamda, float outLamda,
		float alpha_x, float alpha_y) const
	{
		float weight;
		if (bRec.sampledType == EGlossyReflection)
		{
			weight = evalBounce_R_middle(bRec, outside, inLamda, outLamda);
		}
		else
		{
			weight = evalBounce_T_middle(bRec, outside, inLamda, outLamda);
		}
		return weight;
	}

	// during sampling, the other terms are cancelled out.
	float evalBounceSample(BSDFSamplingRecord &bRec, bool outside, float inLamda, float outLamda) const
	{
		float weight;
		if (bRec.sampledType == EGlossyReflection)
		{
			weight = computeG2_R_middle(bRec, outside, inLamda, outLamda);
		}
		else
		{
			weight = computeG2_T_middle(bRec, outside, inLamda, outLamda);

		}
		return weight;
	}


	//we should store more things in this path information, reflect or refract, outside or inside
	float computeWeightSum(const PathSimple &path, int order, BSDFSamplingRecord &bRec) const
	{
		float pdfSum = 0.0f;

		float pdfPre = 1;

		for (int k = 1; k < order - 1; k++)
		{
			pdfPre *= path.invPdf[k - 1];
			float pdfTemp = pdfPre;
			for (int j = k + 1; j < order; j++)
			{
				pdfTemp *= path.invPdf[j];
			}
			pdfSum += pdfTemp;

		}
		return pdfSum;
	}

	Spectrum evalBDPT(BSDFSamplingRecord &bRec,
		int order, EMeasure measure) const
	{
		if (measure != ESolidAngle || Frame::cosTheta(bRec.wi) == 0)
			return Spectrum(0.0f);

		//DONOT MODIFY THESE TWO DIRECTIONS!!!
		Vector w0 = bRec.wi;
		Vector woN = bRec.wo;

		float result(0.0f);
		//	return result;

		PathSample pathForward;
		PathSample pathBack;

		BSDFSamplingRecord bRecTemp = bRec;
		bool reflect = Frame::cosTheta(bRec.wi) * Frame::cosTheta(bRec.wo) > 0;

		float alpha_x = m_alphaU->eval(bRec.its).average();
		float alpha_y = m_alphaV->eval(bRec.its).average();

		samplePath(bRecTemp, order, pathForward, w0.z > 0, alpha_x, alpha_y);

		bRecTemp.wi = woN;
		bRecTemp.wo = w0;

		samplePath_back(bRecTemp, order, pathBack, woN.z > 0, alpha_x, alpha_y);

		//sample count from 1 to order - 1, and the last one is connection

		float outLamda = getLambda(woN.z > 0 ? woN : -woN, alpha_x, alpha_y);
		float inLamda = getInLambda(w0.z > 0 ? w0 : -w0, alpha_x, alpha_y);

		float readyOutLamda = abs(outLamda + 1) - 1;

		PathSimple path;

		for (int k = 0; k < order; k++)
		{
			float resultPathk(1.0);
			float pdfPathK = 1.0F;
			if (k != 0)
			{
				const Vertex &v = pathForward.vList[k - 1];
				if (v.pdf == 0.0)
					break;
				resultPathk *= pathForward.vList[k - 1].weightAcc;
				pdfPathK = pathForward.vList[k - 1].pdfAcc;

			}

			if (resultPathk < THRESHOLD || pdfPathK < THRESHOLD)
				break;

			for (int i = 0; i < order - k; i++)
			{
				if (i != 0)
				{
					const Vertex &vt = pathBack.vList[i - 1];
					if (vt.pdf == 0.0)
						break;
				}

				if (k == 0 && i == 0)
				{

					bRec.sampledType = reflect ? EGlossyReflection : EGlossyTransmission;
					result += evalBounce_last(bRec, w0.z > 0, inLamda, readyOutLamda);
					continue;
				}

				int currentOrder = k + i + 1;
				//k samples for forward path and j samples from back paths

				if (k + i > order - 1)
					continue;

				path.count = 0; //reset the path.
				float pdfForward = 1.0;
				float pdfBackward = 1.0f;

				float resultPath = resultPathk;
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

				if (resultPath < THRESHOLD || pdfPath < THRESHOLD)
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
					bRecTemp.wi = -v.wo;
					bRecTemp.wo = woN;

					bool outside = (v.outside == v.reflect);
					bool reflect = (outside && woN.z > 0) || (!outside &&woN.z < 0);
					float inLamdaTemp = abs(v.outLamda) - 1;

					bRecTemp.sampledType = reflect ? EGlossyReflection : EGlossyTransmission;
					resultPath *= evalBounce_last(bRecTemp, outside, inLamdaTemp, readyOutLamda);

					if (pdfBackward > THRESHOLD)
					{
						bRecTemp.wi = woN;
						bRecTemp.wo = -v.wo;
						float invPdf = pdfWi(bRecTemp, woN.z > 0, readyOutLamda, alpha_x, alpha_y);
						//pdfForward *= pdf;
						pdfBackward *= invPdf;
						float pdf = 1;
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
					bRecTemp.wi = -v_s.wo;// [i];
					bRecTemp.wo = -v_t.wo;// [order - 1 - i - 1];

					bool tOutside = (v_t.outside == v_t.reflect);

					float inLamdaTemp = abs(v_s.outLamda) - 1;
					float outLamdaTemp = abs(v_t.outLamda) - 1;

					if (tOutside == v_s.outside)
						bRecTemp.sampledType = EGlossyReflection;
					else
						bRecTemp.sampledType = EGlossyTransmission;

					resultPath *= evalBounce_middle(bRecTemp, v_s.outside, inLamdaTemp, outLamdaTemp, alpha_x, alpha_y);//  false, false);// / Frame::cosTheta(bRecNew.wi) * Frame::cosTheta(bRecNew.wo);

					float pdf = 0;
					if (pdfForward > THRESHOLD)
					{
						pdf = pdfWi(bRecTemp, v_s.outside, inLamdaTemp, alpha_x, alpha_y);
						pdfForward *= pdf;
					}
					float invPdf = 0;
					if (pdfBackward > THRESHOLD)
					{
						bRecTemp.wo = -v_s.wo;// [i];
						bRecTemp.wi = -v_t.wo;// [order - 1 - i - 1];
						invPdf = pdfWi(bRecTemp, v_t.outside, outLamdaTemp, alpha_x, alpha_y);
						pdfBackward *= invPdf;
					}

					path.add(pdf, invPdf);

				}

				if (k == 0) //this is the first one
				{
					const Vertex &v = pathBack.vList[i - 1];
					bRecTemp.wi = w0;
					bRecTemp.wo = -v.wo;

					float outLamdaTemp = abs(v.outLamda) - 1; //wo is actually wi in backward
					bool outside = (v.outside == v.reflect);

					if ((w0.z > 0 && outside) || (w0.z < 0 && !outside))
						bRecTemp.sampledType = EGlossyReflection;
					else
						bRecTemp.sampledType = EGlossyTransmission;

					resultPath *= evalBounce_middle(bRecTemp, w0.z > 0, inLamda, outLamdaTemp, alpha_x, alpha_y);

					float invPdf = 1;
					pdfBackward *= invPdf;

					float pdf = 0;
					if (pdfForward > THRESHOLD)
					{
						pdf = pdfWi(bRecTemp, w0.z > 0, inLamda, alpha_x, alpha_y);
						pdfForward *= pdf;
					}
					path.add(pdf, invPdf);
				}

				for (int j = i - 1; j >= 0; j--)
				{
					const Vertex &v = pathBack.vList[j];

					bool outside = (v.outside == v.reflect);

					path.add(v.pdf, v.invPdf);
				}

				if (resultPath < THRESHOLD || pdfPath <THRESHOLD)
					continue;


				float pdfSum =	pdfForward + pdfBackward + computeWeightSum(path, currentOrder, bRecTemp);

				if (!IsFiniteNumber(pdfSum))
				{
					return Spectrum(0.0f);
				}
				result += pdfSum == 0.0 ? (0.0f) : resultPath * pdfPath / pdfSum;
			}

		}

		if (!reflect)
		{
			Float factor = (bRec.mode == ERadiance)
				? (w0.z > 0 ? m_invEta : m_eta) : 1.0f;
			result *= factor*factor;
		}

		return Spectrum(result);
	}


	/*our unidirectional estimator for the path integral*/
	Spectrum evalPT(BSDFSamplingRecord &_bRec,
		int order, EMeasure measure) const
	{
		if (measure != ESolidAngle || Frame::cosTheta(_bRec.wi) == 0)
			return Spectrum(0.0f);

		BSDFSamplingRecord bRec = _bRec;
		Vector w0 = bRec.wi;
		Vector woN = bRec.wo;
		float result(0.0f);

		/* Determine the type of interaction */
		bool outside = w0.z > 0;
		float alpha_x = m_alphaU->eval(bRec.its).average();
		float alpha_y = m_alphaV->eval(bRec.its).average();
		float inLamda = getInLambda(w0.z > 0 ? w0 : -w0, alpha_x, alpha_y);
		float outLamda = getLambda(woN.z > 0 ? woN : -woN, alpha_x, alpha_y);
		float readyOutLamda = abs(outLamda + 1) - 1;

		bool reflect = Frame::cosTheta(bRec.wi)	* Frame::cosTheta(bRec.wo) > 0;
		bRec.sampledType = reflect ? EGlossyReflection : EGlossyTransmission;
		result += evalBounce_last(bRec, outside, inLamda, readyOutLamda);

		float weight = (1.0f);
		float pdf = 1;
		float eta = 1.0f;

		for (int i = 0; i < order - 1; i++)
		{
			//sampling a direcction with the normal distribution function
			float currentPDF;
			sampleVNDF(bRec, currentPDF, outside, inLamda, alpha_x, alpha_y);
			eta *= bRec.eta;

			if (currentPDF == 0.0f) break;

			pdf *= currentPDF;
			bool newOutside = outside == (bRec.sampledType == EGlossyReflection);
			float newOutLamda = getLambda(newOutside ? bRec.wo : -bRec.wo, alpha_x, alpha_y);

			weight *= evalBounceSample(bRec, outside, inLamda, abs(newOutLamda + 1) - 1);

			outside = newOutside;

			if (weight / pdf < 1e-4 || pdf < 1e-4) break;

			bRec.wi = -bRec.wo;
			bRec.wo = woN;
			Vector tempWo = newOutside ? bRec.wi : -bRec.wi;
			inLamda = abs(newOutLamda) - 1;

			float currentWight;
			if (outside && bRec.wo.z > 0 || (!outside && bRec.wo.z < 0))
				currentWight = weight * evalBounce_R_last(bRec, outside, inLamda, readyOutLamda); 
			else
				currentWight = weight * evalBounce_T_last(bRec, outside, inLamda, readyOutLamda);

			result += currentWight / pdf;

			if (i + 1 >= m_rrDepth) {
				Float q = std::min(currentWight * eta * eta, (Float) 0.95f);
				if (bRec.sampler->next1D() >= q)
					break;
				currentWight /= q;
			}
		}

		if (!reflect)
		{
			Float factor = (bRec.mode == ERadiance)
				? (w0.z > 0 ? m_invEta : m_eta) : 1.0f;
			result *= factor*factor;
		}

		return Spectrum(result);
	}

	//vertex term for the reflection, including the specular reflectance
	inline Spectrum computeD_F_S(const BSDFSamplingRecord &bRec, bool outside) const
	{
		if ((bRec.wo + bRec.wi).isZero())
			return Spectrum(0.0f);

		/* Calculate the reflection half-vector */
		Vector H = normalize(bRec.wo + bRec.wi);
		/* Ensure that the half-vector points into the
		same hemisphere as the macrosurface normal */
		H *= math::signum(Frame::cosTheta(H));

		if (outside && (dot(H, bRec.wi) <= 0 || dot(H, bRec.wo) <= 0))
			return Spectrum(0.0f);

		if (!outside && (dot(H, bRec.wi) >= 0 || dot(H, bRec.wo) >= 0))
			return Spectrum(0.0f);

		if (H.z <= 0.0f)
			return  Spectrum(0.0f);

		float D = m_distr->eval(H);
		if (D == 0)
			return  Spectrum(0.0f);

		/* Fresnel factor */
		const Spectrum F = fresnelDielectricExt(dot(bRec.wi, H), m_eta) * m_specularReflectance->eval(bRec.its);

		return D * F / (4 * abs(bRec.wi.z));
	}

	//vertex term for the transmission, do not the specular transmission
	inline float computeD_F_T(const BSDFSamplingRecord &bRec, bool outside) const
	{
	
		Float eta = outside	? m_eta : m_invEta;

		Vector H = normalize(bRec.wi + bRec.wo*eta);
		H *= math::signum(Frame::cosTheta(H));

		if (outside && (dot(H, bRec.wi) <= 0 || dot(H, bRec.wo) >= 0))
			return (0.0f);

		if (!outside && (dot(H, bRec.wi) >= 0 || dot(H, bRec.wo) <= 0))
			return (0.0f);

		float D = m_distr->eval(H);

		if (D == 0)
			return  (0.0f);

		/* Smith's shadow-masking function  distr.smithG1(bRec.wo, H)*/
		if (dot(bRec.wo, H) * Frame::cosTheta(bRec.wo) <= 0)
			return (0.0f);

		const float F = fresnelDielectricExt(dot(bRec.wi, H), m_eta);//

		Float sqrtDenom = dot(bRec.wi, H) + eta * dot(bRec.wo, H);

		Float value = ((1 - F) *  D * dot(bRec.wo, H)*dot(bRec.wi, H) * eta * eta) / (abs(bRec.wi.z) * sqrtDenom * sqrtDenom);
		return abs(value);
	}

	//vertex term for the reflection, without the specular reflectance
	inline float computeD_F(const BSDFSamplingRecord &bRec, bool outside) const
	{
		if ((bRec.wo + bRec.wi).isZero())
			return (0.0f);

		Vector H = normalize(bRec.wo + bRec.wi);
		H *= math::signum(Frame::cosTheta(H));

		if (outside && (dot(H, bRec.wi) <= 0 || dot(H, bRec.wo) <= 0))
			return (0.0f);

		if (!outside && (dot(H, bRec.wi) >= 0 || dot(H, bRec.wo) >= 0))
			return (0.0f);

		if (H.z <= 0.0f)
			return  (0.0f);

		float D = m_distr->eval(H);
		if (D == 0)
			return  (0.0f);

		/* Fresnel factor */
		const float F = fresnelDielectricExt(dot(bRec.wi, H), m_eta);
		return D * F / (4 * abs(bRec.wi.z));
	}



	Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const {
		if (measure != ESolidAngle)
			return 0.0f;

		/* Determine the type of interaction */
		bool hasReflection = ((bRec.component == -1 || bRec.component == 0)
			&& (bRec.typeMask & EGlossyReflection)),
			hasTransmission = ((bRec.component == -1 || bRec.component == 1)
			&& (bRec.typeMask & EGlossyTransmission)),
			reflect = Frame::cosTheta(bRec.wi)
			* Frame::cosTheta(bRec.wo) > 0;

		Vector wh;
		Float dwh_dwo;

		if (reflect) {
			/* Zero probability if this component was not requested */
			if ((bRec.component != -1 && bRec.component != 0)
				|| !(bRec.typeMask & EGlossyReflection))
				return 0.0f;

			/* Calculate the reflection half-vector */
			wh = normalize(bRec.wo + bRec.wi);

			/* Jacobian of the half-direction mapping */
			dwh_dwo = 1.0f / (4.0f * dot(bRec.wo, wh));
		}
		else {
			/* Zero probability if this component was not requested */
			if ((bRec.component != -1 && bRec.component != 1)
				|| !(bRec.typeMask & EGlossyTransmission))
				return 0.0f;

			/* Calculate the transmission half-vector */
			Float eta = Frame::cosTheta(bRec.wi) > 0
				? m_eta : m_invEta;

			wh = normalize(bRec.wi + bRec.wo*eta);

			/* Jacobian of the half-direction mapping */
			Float sqrtDenom = dot(bRec.wi, wh) + eta * dot(bRec.wo, wh);
			dwh_dwo = (eta*eta * dot(bRec.wo, wh)) / (sqrtDenom*sqrtDenom);
		}

		/* Ensure that the half-vector points into the
		same hemisphere as the macrosurface normal */
		float alpha_x = m_alphaU->eval(bRec.its).average();
		float alpha_y = m_alphaV->eval(bRec.its).average();
		MicrofacetDistribution distr(
			m_type,
			alpha_x,
			alpha_y,
			m_sampleVisible
			);

		wh *= math::signum(Frame::cosTheta(wh));
		//	Float s = math::signum(Frame::cosTheta(bRec.wi));
		//	float G1 = computeG1(s * bRec.wi, alpha_x, alpha_y);
		//	Float prob = std::max(0.0f, dot(wh, bRec.wi)) * distr.eval(wh) * G1 / Frame::cosTheta(bRec.wi);

		Float prob = distr.pdf(math::signum(Frame::cosTheta(bRec.wi)) * bRec.wi, wh);
		if (hasTransmission && hasReflection) {
			Float F = fresnelDielectricExt(dot(bRec.wi, wh), m_eta);
			prob *= reflect ? F : (1 - F);
		}

		// single-scattering PDF + diffuse 
		// otherwise too many fireflies due to lack of multiple-scattering PDF
		// (MIS works even if the PDF is wrong and not normalized)
		return std::abs(prob/* * s*/ * dwh_dwo) + Frame::cosTheta(bRec.wo);
	}

	Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &_sample) const {
		float pdf;
		return this->sample(bRec, pdf, _sample);
	}

	Spectrum sample(BSDFSamplingRecord &bRec, Float &pdf, const Point2 &_sample) const {
		//return sample_microfacet(bRec, pdf, _sample);
		Point2 sample(_sample);

		const int order = m_order; //1;// 
		bool hasReflection = ((bRec.component == -1 || bRec.component == 0)
			&& (bRec.typeMask & EGlossyReflection)),
			hasTransmission = ((bRec.component == -1 || bRec.component == 1)
			&& (bRec.typeMask & EGlossyTransmission)),
			sampleReflection = hasReflection;

		if (!hasReflection && !hasTransmission)
			return Spectrum(0.0f);
		float alpha_x = m_alphaU->eval(bRec.its).average();
		float alpha_y = m_alphaV->eval(bRec.its).average();
		Spectrum weight(1.0f);
		pdf = 1.0f;
		int bounce = 0;
		Vector w0 = bRec.wi;
		bool outside = w0.z > 0;
		float inLamda = getInLambda(outside ? w0 : -w0, alpha_x, alpha_y);

		while (true)
		{
			float currentPdf;

			sampleVNDF(bRec, currentPdf, outside, inLamda, alpha_x, alpha_y);

			pdf *= currentPdf;

			if (pdf < 1e-10)
				return Spectrum(0.0f);

			bool curOutside = ((bRec.sampledType == EGlossyReflection) ? outside : !outside);
			bounce++;
			bool needG = curOutside ? Frame::cosTheta(bRec.wo) > 0 : Frame::cosTheta(bRec.wo) < 0;
			const Vector wo = curOutside ? bRec.wo : -bRec.wo;
			float outLamda = getLambda(wo, alpha_x, alpha_y);
			if (needG)
			{

#if G_REUSE
				float G1 = 1 / abs(1 + outLamda);
#else
				float G1 = computeG1(wo);
#endif

				float rand = bRec.sampler->next1D();

				if (rand < G1)
				{
					pdf *= G1;
					outside = curOutside;
					break;
				}
				else
				{
					pdf *= (1 - G1);
				}

			}
			bRec.wi = -bRec.wo;
			inLamda = abs(outLamda) - 1;
			outside = curOutside;

			if (bounce >= order)
			{
				pdf = 0.0f;
				weight = Spectrum(0.0f);
				bRec.wo = Vector3(0, 0, 1);
				break;
			}

		}
		bRec.wi = w0;

		pdf = this->pdf(bRec, ESolidAngle);

		bool initialOutside = w0.z > 0;
		if ((initialOutside && outside) || (!initialOutside && !outside))
		{
			bRec.eta = 1.0f;
			bRec.sampledComponent = 0;
			bRec.sampledType = EGlossyReflection;
			const Spectrum R = m_specularReflectance->eval(bRec.its);

			/* Side check */
			if (Frame::cosTheta(bRec.wi) * Frame::cosTheta(bRec.wo) <= 0)
				return Spectrum(0.0f);
			return R * weight;

		}
		else
		{
			bRec.eta = initialOutside ? m_eta : m_invEta;
			bRec.sampledComponent = 1;
			bRec.sampledType = EGlossyTransmission;

			const Spectrum T = m_specularTransmittance->eval(bRec.its);
			Float factor = (bRec.mode == ERadiance)
				? (initialOutside ? m_invEta : m_eta) : 1.0f;

			//	printLog("weightT", factor * factor * weight[0]);
			return T * factor * factor * weight;
		}
	}

	void addChild(const std::string &name, ConfigurableObject *child) {
		if (child->getClass()->derivesFrom(MTS_CLASS(Texture))) {
			if (name == "alpha")
				m_alphaU = m_alphaV = static_cast<Texture *>(child);
			else if (name == "alphaU")
				m_alphaU = static_cast<Texture *>(child);
			else if (name == "alphaV")
				m_alphaV = static_cast<Texture *>(child);
			else if (name == "specularReflectance")
				m_specularReflectance = static_cast<Texture *>(child);
			else if (name == "specularTransmittance")
				m_specularTransmittance = static_cast<Texture *>(child);
			else
				BSDF::addChild(name, child);
		}
		else {
			BSDF::addChild(name, child);
		}
	}

	Float getEta() const {
		return m_eta;
	}

	Float getRoughness(const Intersection &its, int component) const {
		return 0.5f * (m_alphaU->eval(its).average()
			+ m_alphaV->eval(its).average());
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

	std::string toString() const {
		std::ostringstream oss;
		oss << "RoughDielectric[" << endl
			<< "  id = \"" << getID() << "\"," << endl
			<< "  distribution = " << MicrofacetDistribution::distributionName(m_type) << "," << endl
			<< "  sampleVisible = " << m_sampleVisible << "," << endl
			<< "  eta = " << m_eta << "," << endl
			<< "  alphaU = " << indent(m_alphaU->toString()) << "," << endl
			<< "  alphaV = " << indent(m_alphaV->toString()) << "," << endl
			<< "  specularReflectance = " << indent(m_specularReflectance->toString()) << "," << endl
			<< "  specularTransmittance = " << indent(m_specularTransmittance->toString()) << endl
			<< "]";
		return oss.str();
	}

	Shader *createShader(Renderer *renderer) const;

	MTS_DECLARE_CLASS()
private:
	MicrofacetDistribution::EType m_type;
	ref<Texture> m_specularTransmittance;
	ref<Texture> m_specularReflectance;
	ref<Texture> m_alphaU, m_alphaV;
	Float m_eta, m_invEta;
	bool m_sampleVisible;

	//for mbndf
	int m_order;
	int m_rrDepth;

	bool m_bdpt;
	bool m_hCorrelated;

	float m_alpha_x;
	float m_alpha_y;
	MicrofacetDistribution *m_distr;

};

/* Fake glass shader -- it is really hopeless to visualize
this material in the VPL renderer, so let's try to do at least
something that suggests the presence of a transparent boundary */
class RoughDielectricShader : public Shader {
public:
	RoughDielectricShader(Renderer *renderer, Float eta) :
		Shader(renderer, EBSDFShader) {
		m_flags = ETransparent;
	}

	Float getAlpha() const {
		return 0.3f;
	}

	void generateCode(std::ostringstream &oss,
		const std::string &evalName,
		const std::vector<std::string> &depNames) const {
		oss << "vec3 " << evalName << "(vec2 uv, vec3 wi, vec3 wo) {" << endl
			<< "    if (cosTheta(wi) < 0.0 || cosTheta(wo) < 0.0)" << endl
			<< "    	return vec3(0.0);" << endl
			<< "    return vec3(inv_pi * cosTheta(wo));" << endl
			<< "}" << endl
			<< endl
			<< "vec3 " << evalName << "_diffuse(vec2 uv, vec3 wi, vec3 wo) {" << endl
			<< "    return " << evalName << "(uv, wi, wo);" << endl
			<< "}" << endl;
	}


	MTS_DECLARE_CLASS()
};

Shader *RoughDielectric::createShader(Renderer *renderer) const {
	return new RoughDielectricShader(renderer, m_eta);
}

MTS_IMPLEMENT_CLASS(RoughDielectricShader, false, Shader)
MTS_IMPLEMENT_CLASS_S(RoughDielectric, false, BSDF)
MTS_EXPORT_PLUGIN(RoughDielectric, "Rough dielectric BSDF");
MTS_NAMESPACE_END
