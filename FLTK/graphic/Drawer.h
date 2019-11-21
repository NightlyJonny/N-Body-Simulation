#pragma once
#ifndef DRAW_HPP
#define DRAW_HPP

#include <FL/gl.h>
#include <FL/glu.h>
#include <GL/glut.h>	  
#include <math.h>
#include "../Simulation.h"

class Drawer {

private:
	Particle* particle;
	int NParticle;
public:
	Drawer(Simulation* sim);
	void drawFilledCircle(GLfloat x, GLfloat y, GLfloat radius);
	void draw_scene();

};

#endif // DRAW_HPP
