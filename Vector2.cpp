#include "Vector2.h"
#include <cmath>

Vector2::Vector2 () : x(0), y(0) {}

Vector2::Vector2 (double x0, double y0) : x(x0), y(y0) {}

Vector2::~Vector2 () {}

Vector2 Vector2::operator+ (const Vector2& v2) { return Vector2(x + v2.x, y + v2.y); }

Vector2 Vector2::operator- (const Vector2& v2) { return Vector2(x - v2.x, y - v2.y); }

Vector2 Vector2::operator* (double s) { return Vector2(x * s, y * s); }

double Vector2::operator* (const Vector2& v2) { return x*v2.x + y*v2.y; }

Vector2 Vector2::operator/ (double s) { return Vector2(x / s, y / s); }

double Vector2::norm () const {
	return sqrt(x*x + y*y);
}

Vector2 Vector2::versor () const {
	return Vector2(x, y) / norm();
}