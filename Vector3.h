#ifndef VECTOR3_H
#define VECTOR3_H

class Vector3 {
public:
	float x, y, z;
	Vector3 ();
	Vector3 (float, float, float);
	~Vector3 ();

	Vector3 operator+ (const Vector3&);
	Vector3 operator- (const Vector3&);
	Vector3 operator* (float);
	float operator* (const Vector3&);
	Vector3 operator/ (float);
	Vector3 cross (const Vector3&) const;

	float norm () const;
	Vector3 versor () const;
};

#endif /* VECTOR3_H */