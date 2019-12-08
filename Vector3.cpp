#include "Vector3.h"
#include <cmath>

Vector3::Vector3 () : x(0), y(0), z(0) {}

Vector3::Vector3 (float x0, float y0, float z0) : x(x0), y(y0), z(z0) {}

Vector3::~Vector3 () {}

Vector3 Vector3::operator+ (const Vector3& v2) { return Vector3(x + v2.x, y + v2.y, z + v2.z); }

Vector3 Vector3::operator- (const Vector3& v2) { return Vector3(x - v2.x, y - v2.y, z - v2.z); }

Vector3 Vector3::operator* (float s) { return Vector3(x * s, y * s, z * s); }

float Vector3::operator* (const Vector3& v2) { return x*v2.x + y*v2.y + z*v2.z; }

Vector3 Vector3::operator/ (float s) { return Vector3(x / s, y / s, z / s); }

Vector3 Vector3::cross (const Vector3& v2) const {

	return Vector3(y*v2.z - z*v2.y, z*v2.x - x*v2.z, x*v2.y - y*v2.x);
}

float Vector3::norm () const {

	return sqrt(x*x + y*y + z*z);
}

Vector3 Vector3::versor () const {

	return Vector3(x, y, z) / norm();
}