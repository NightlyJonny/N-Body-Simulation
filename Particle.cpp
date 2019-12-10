#include "Particle.h"

Particle::Particle () : mass(1), radius(0.1), position(), velocity(), angle(0), omega() {}

Particle::Particle (float m, float r) : mass(m), radius(r), position(), velocity(), angle(0), omega() {};

Particle::Particle (float m, float r, Vector3 p, Vector3 v, float a, Vector3 o) : mass(m), radius(r), position(p.x, p.y, p.z), velocity(v.x, v.y, v.z), angle(a), omega(o.x, o.y, o.z) {};

Particle::~Particle () {}

void Particle::initialize () {
	Vector3 rpVector (random(-20, 20), random(-20, 20), random(-20, 20));
	position = rpVector;
	omega = Vector3(random(-1, 1), random(-1, 1), random(-1, 1));
	mass = random(0.5, 2);
	radius = random(0.1, 0.2);
}

void Particle::energyInitialize (float R, float E0) {
	const float MAXMASS = 2.0;
	const float MINMASS = 0.5;
	Vector3 rpVector;
	do {
		rpVector = Vector3(random(-R, R), random(-R, R), random(-R, R));
	} while (rpVector.norm() > R);
	position = rpVector;
	mass = random(MINMASS, MAXMASS);
	radius = random(0.1, 0.2);

	float v = sqrt( 4 * E0 / (MAXMASS - MINMASS) + 199 * 35 * (MAXMASS - MINMASS) / (72 * R) );
	velocity = position.versor() * random(0, 2*v);
}

void Particle::momentumInitialize (float L0) {
	initialize();
}

void Particle::positionStep (float dt, Vector3& force) {

	position = position + velocity * dt;
}

void Particle::velocityStep (float dt, Vector3& force) {
	
	velocity = velocity + (force/mass) * dt;
}

void Particle::angularStep (float dt) {
	
	angle = fmod(angle + omega.norm() * dt, 2*PI);
}