#include "Particle.h"

Particle::Particle () : mass(1), radius(0.1), position(0, 0), velocity(0, 0), angle(0), omega(0) {}

Particle::Particle (double m, double r) : mass(m), radius(r), position(), velocity(), angle(0), omega(0) {};

Particle::Particle (double m, double r, Vector2 p, Vector2 v, double a, double o) : mass(m), radius(r), position(p.x, p.y), velocity(v.x, v.y), angle(a), omega(o) {};

Particle::~Particle () {}

void Particle::initialize () {
	Vector2 rpVector (random(-40, 40), random(-40, 40));
	Vector2 tpVector (-rpVector.y, rpVector.x);
	position = rpVector;
	velocity = rpVector.versor() * random(-3, 3) + tpVector.versor() * random(0, 8);
	omega = random(-1, 1);
	mass = random(0.5, 2);
	radius = random(0.1, 0.2);
}

void Particle::positionStep (double dt, Vector2& force) {
	position = position + velocity * dt;
	// position = position + velocity * dt + force * 0.5 * dt*dt;
}

void Particle::velocityStep (double dt, Vector2& force) {
	velocity = velocity + (force/mass) * dt;
	// velocity = velocity + force * dt;
}

void Particle::angularStep (double dt) {
	
	angle = remainder(angle + omega * dt, 2*PI);
}