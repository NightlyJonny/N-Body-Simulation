#ifndef PARTICLE_H
#define PARTICLE_H

#include "Vector2.h"
#include <cmath>
#define PI 3.14159265359

class Particle {
private:
	double remainder(double a, double b) { return a - (int)(a/b) * b; }

public:
	double mass, radius, angle, omega;
	Vector2 position, velocity;
	Particle ();
	Particle (double, double);
	Particle (double, double, Vector2, Vector2, double, double);
	~Particle ();
	
	void positionStep (double, Vector2&);
	void velocityStep (double, Vector2&);
	void angularStep (double);
};

#endif /* PARTICLE_H */