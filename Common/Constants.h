#pragma once

namespace JEngine
{
	template<typename T>
	constexpt T PI = (T)3.141592653589793L;

	template<typename T>
	constexpt T TWO_PI = 2 * PI<T>;

	template<typename T>
	constexpt T PI_PI = PI<T> * PI<T>;

	template<typename T>
	constexpt T DEG_TO_RAD = T(180) / PI<T>;

	template<typename T>
	constexpt T RAD_TO_DEG = PI<T> / T(180);

	template<typename T>
	T DegreeToRad(T degree) { return degree * DEG_TO_RAD; }

	template<typename T>
	T RadToDegree(T rad) { return rad * RAD_TO_DEG; }
}