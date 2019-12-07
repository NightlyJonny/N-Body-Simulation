#include "Vector3.h"
#include <cmath>

Vector3::Vector3 () : x(0), y(0), z(0) {}

Vector3::Vector3 (double x0, double y0, double z0) : x(x0), y(y0), z(z0) {}

Vector3::~Vector3 () {}

Vector3 Vector3::operator+ (const Vector3& v2) { return Vector3(x + v2.x, y + v2.y, z + v2.z); }

Vector3 Vector3::operator- (const Vector3& v2) { return Vector3(x - v2.x, y - v2.y, z - v2.z); }

Vector3 Vector3::operator* (double s) { return Vector3(x * s, y * s, z * s); }

double Vector3::operator* (const Vector3& v2) { return x*v2.x + y*v2.y + z*v2.z; }

Vector3 Vector3::operator/ (double s) { return Vector3(x / s, y / s, z / s); }

double Vector3::norm () const {
	return sqrt(x*x + y*y + z*z);
}

Vector3 Vector3::versor () const {
	return Vector3(x, y, z) / norm();
}