#include "Particle.h"

Particle::Particle () : mass(1), radius(0.1), position(), velocity(), angle(0), omega() {}

Particle::Particle (float m, float r) : mass(m), radius(r), position(), velocity(), angle(0), omega() {};

Particle::Particle (float m, float r, Vector3 p, Vector3 v, float a, Vector3 o) : mass(m), radius(r), position(p.x, p.y, p.z), velocity(v.x, v.y, v.z), angle(a), omega(o.x, o.y, o.z) {};

Particle::~Particle () {}

void Particle::positionStep (float dt, Vector3& force) {

	position = position + velocity * dt;
}

void Particle::velocityStep (float dt, Vector3& force) {
	
	velocity = velocity + (force/mass) * dt;
}

void Particle::angularStep (float dt) {
	
	angle = fmod(angle + omega.norm() * dt, 2*PI);
}