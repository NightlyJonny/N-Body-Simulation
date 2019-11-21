#ifndef PARTICLE_H
#define PARTICLE_H

#include "Vector2.h"

class Particle {
public:
	double mass, radius;
	Vector2 position, velocity;
	Particle ();
	Particle (double, double);
	Particle (double, double, Vector2, Vector2);
	~Particle ();
	
	void positionStep (double, Vector2&);
	void velocityStep (double, Vector2&);
};

#endif /* PARTICLE_H */