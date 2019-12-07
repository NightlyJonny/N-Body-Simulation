#ifndef VECTOR3_H
#define VECTOR3_H

class Vector3 {
public:
	double x, y, z;
	Vector3 ();
	Vector3 (double, double, double);
	~Vector3 ();

	Vector3 operator+ (const Vector3&);
	Vector3 operator- (const Vector3&);
	Vector3 operator* (double);
	double operator* (const Vector3&);
	Vector3 operator/ (double);

	double norm () const;
	Vector3 versor () const;
};

#endif /* VECTOR3_H */