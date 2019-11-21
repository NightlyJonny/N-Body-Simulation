#include "Particle.h"

__host__ __device__ Particle::Particle () : mass(1), radius(0.1), position(0, 0), velocity(0, 0) {}

__host__ __device__ Particle::Particle (double m, double r) : mass(m), radius(r), position(), velocity() {};

__host__ __device__ Particle::Particle (double m, double r, Vector2 p, Vector2 v) : mass(m), radius(r), position(p.x, p.y), velocity(v.x, v.y) {};

__host__ __device__ Particle::~Particle () {}

__host__ __device__ void Particle::step (double dt, Vector2 force) {
	position = position + velocity * dt + force * 0.5 * dt*dt;
	velocity = velocity + force * dt;
}