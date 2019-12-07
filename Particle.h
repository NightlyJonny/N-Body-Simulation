#ifndef PARTICLE_H
#define PARTICLE_H

#include "Vector3.h"
#include <cmath>
#define PI 3.14159265359

class Particle {
private:
	double random(double min, double max) { return ((double)rand() / RAND_MAX) * (max - min) + min; }

public:
	bool active = true;
	double mass, radius, angle, omega;
	Vector3 position, velocity;
	Particle ();
	Particle (double, double);
	Particle (double, double, Vector3, Vector3, double, double);
	~Particle ();
	
	void positionStep (double, Vector3&);
	void velocityStep (double, Vector3&);
	void angularStep (double);
	void initialize ();
};

#endif /* PARTICLE_H */