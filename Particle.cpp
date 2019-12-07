#include "Particle.h"

Particle::Particle () : mass(1), radius(0.1), position(), velocity(), angle(0), omega() {}

Particle::Particle (double m, double r) : mass(m), radius(r), position(), velocity(), angle(0), omega() {};

Particle::Particle (double m, double r, Vector3 p, Vector3 v, double a, Vector3 o) : mass(m), radius(r), position(p.x, p.y, p.z), velocity(v.x, v.y, v.z), angle(a), omega(o.x, o.y, o.z) {};

Particle::~Particle () {}

void Particle::initialize () {
	Vector3 rpVector (random(-20, 20), random(-20, 20), random(-20, 20));
	position = rpVector;
	omega = Vector3(random(-1, 1), random(-1, 1), random(-1, 1));
	mass = random(0.5, 2);
	radius = random(0.1, 0.2);
}

void Particle::positionStep (double dt, Vector3& force) {

	position = position + velocity * dt;
}

void Particle::velocityStep (double dt, Vector3& force) {
	
	velocity = velocity + (force/mass) * dt;
}

void Particle::angularStep (double dt) {
	
	angle = fmod(angle + omega.norm() * dt, 2*PI);
}