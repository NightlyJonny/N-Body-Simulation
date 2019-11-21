#ifndef PARTICLE_H
#define PARTICLE_H

#include "Vector2.h"

class Particle {
private:
	double mass, radius;

public:
	Vector2 position, velocity;
	__host__ __device__ Particle ();
	__host__ __device__ Particle (double, double);
	__host__ __device__ Particle (double, double, Vector2, Vector2);
	__host__ __device__ ~Particle ();

	__host__ __device__ double getMass () { return mass; };
	__host__ __device__ double getRadius () { return radius; };
	__host__ __device__ void step (double, Vector2);
};

#endif /* PARTICLE_H */