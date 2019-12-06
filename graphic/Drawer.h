#pragma once
#ifndef DRAW_HPP
#define DRAW_HPP

#include <FL/gl.h>
#include <FL/glu.h>
#include <GL/glut.h>	  
#include <math.h>
#include "../Simulation.h"
#define PI 3.14159265359

class Drawer {

private:
	Particle* particle;
	int NParticle;
	bool asteroids = false;
	GLfloat angle = 0, x = 0, y = 0;
	//This is useful for the creation of the cylinder
	GLUquadricObj *quadratic;

public:
	Drawer();
	void drawSphere(GLfloat x, GLfloat y, GLfloat z, GLfloat radius, GLfloat angle, GLfloat xRot, GLfloat yRot, GLfloat zRot);
	void draw_scene();
	void setSimulation(Simulation* sim);

	void setAsteroids(bool active);
	void drawSpaceship();
	void rotateSpaceship(GLfloat angle);
	void moveSpaceship(GLfloat ds);
	bool shooting = false;
	void shoot();
};

#endif // DRAW_HPP
