#include "Vector2.h"
#include <cmath>

__host__ __device__ Vector2::Vector2 () : x(0), y(0) {}

__host__ __device__ Vector2::Vector2 (double x0, double y0) : x(x0), y(y0) {}

__host__ __device__ Vector2::~Vector2 () {}

__host__ __device__ Vector2 Vector2::operator+ (const Vector2& v2) { return Vector2(x + v2.x, y + v2.y); }

__host__ __device__ Vector2 Vector2::operator- (const Vector2& v2) { return Vector2(x - v2.x, y - v2.y); }

__host__ __device__ Vector2 Vector2::operator* (double s) { return Vector2(x * s, y * s); }

__host__ __device__ double Vector2::operator* (const Vector2& v2) { return x*v2.x + y*v2.y; }

__host__ __device__ Vector2 Vector2::operator/ (double s) { return Vector2(x / s, y / s); }

__host__ __device__ double Vector2::norm () {
	return sqrt(x*x + y*y);
}

__host__ __device__ Vector2 Vector2::versor () {
	return (*this) / this->norm();
}