#pragma once

namespace JEngine
{
	template<typename T>
	constexpr T PI = (T)3.141592653589793L;

	template<typename T>
	constexpr T HALF_PI = PI<T> *(T)0.5;

	template<typename T>
	constexpr T TWO_PI = PI<T> *(T)2;

	template<typename T>
	constexpr T PI_PI = PI<T> *PI<T>;

	template<typename T>
	constexpr T DEG_TO_RAD = (T)180 / PI<T>;

	template<typename T>
	T DegreeToRad(T degree) { return degree * DEG_TO_RAD<T>; }

	template<typename T>
	T RadToDegree(T rad) { return rad / DEG_TO_RAD<T>; }

}