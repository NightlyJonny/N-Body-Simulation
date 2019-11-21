#ifndef VECTOR2_H
#define VECTOR2_H

class Vector2 {
public:
	double x, y;
	__host__ __device__ Vector2 ();
	__host__ __device__ Vector2 (double, double);
	__host__ __device__ ~Vector2 ();

	__host__ __device__ Vector2 operator+ (const Vector2&);
	__host__ __device__ Vector2 operator- (const Vector2&);
	__host__ __device__ Vector2 operator* (double);
	__host__ __device__ double operator* (const Vector2&);
	__host__ __device__ Vector2 operator/ (double);

	__host__ __device__ double norm ();
	__host__ __device__ Vector2 versor ();
};

#endif /* VECTOR2_H */