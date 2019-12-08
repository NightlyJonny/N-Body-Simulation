#ifndef PARTICLE_H
#define PARTICLE_H

#include "Vector3.h"
#include <cmath>
#define PI 3.14159265359

class Particle {
private:
	float random(float min, float max) { return ((float)rand() / RAND_MAX) * (max - min) + min; }

public:
	bool active = true;
	float mass, radius, angle;
	Vector3 position, velocity, omega;
	Particle ();
	Particle (float, float);
	Particle (float, float, Vector3, Vector3, float, Vector3);
	~Particle ();
	
	void positionStep (float, Vector3&);
	void velocityStep (float, Vector3&);
	void angularStep (float);
	void initialize ();
};

#endif /* PARTICLE_H */