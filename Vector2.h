#ifndef VECTOR2_H
#define VECTOR2_H

class Vector2 {
public:
	double x, y;
	Vector2 ();
	Vector2 (double, double);
	~Vector2 ();

	Vector2 operator+ (const Vector2&);
	Vector2 operator- (const Vector2&);
	Vector2 operator* (double);
	double operator* (const Vector2&);
	Vector2 operator/ (double);

	double norm () const;
	Vector2 versor () const;
};

#endif /* VECTOR2_H */