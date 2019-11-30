#ifndef PARTICLE_H
#define PARTICLE_H

#include "Vector2.h"
#include <cmath>
#define PI 3.14159265359

class Particle {
private:
	double remainder(double a, double b) { return a - (int)(a/b) * b; }
	double random(double min, double max) { return ((double)rand() / RAND_MAX) * (max - min) + min; }

public:
	bool active = true;
	double mass, radius, angle, omega;
	Vector2 position, velocity;
	Particle ();
	Particle (double, double);
	Particle (double, double, Vector2, Vector2, double, double);
	~Particle ();
	
	void positionStep (double, Vector2&);
	void velocityStep (double, Vector2&);
	void angularStep (double);
	void initialize ();
};

#endif /* PARTICLE_H */